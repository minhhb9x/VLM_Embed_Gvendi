import os
import io
from typing import Dict, Tuple, Optional
import time
import json
import pickle
import math
from datasets import load_dataset, concatenate_datasets
import torch
import torch.nn as nn
import PIL
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from src.model.model import MMEBModel
from src.model.processor import VLM_IMAGE_TOKENS, load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, PHI3V
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 
from transformers import ProcessorMixin
from qwen_vl_utils import smart_resize
from PIL import Image

POS_MOD_CLASS_LABEL = "Represent the class label: "
POS_MOD_IMAGE_CAPTION = "Represent the image caption: "
POS_MOD_ANSWER = "Represent the answer: "

POS_MOD_DICT = {
                "ImageNet_1K": POS_MOD_CLASS_LABEL,"HatefulMemes":POS_MOD_CLASS_LABEL,"SUN397":POS_MOD_CLASS_LABEL,"N24News":POS_MOD_CLASS_LABEL,"VOC2007":POS_MOD_CLASS_LABEL, "Place365":POS_MOD_CLASS_LABEL,"ImageNet-A":POS_MOD_CLASS_LABEL,"ImageNet-R":POS_MOD_CLASS_LABEL,"ObjectNet":POS_MOD_CLASS_LABEL,"Country211":POS_MOD_CLASS_LABEL,
                
                "OK-VQA":POS_MOD_ANSWER, "A-OKVQA":POS_MOD_ANSWER, "DocVQA":POS_MOD_ANSWER, "InfographicsVQA":POS_MOD_ANSWER, "ChartQA":POS_MOD_ANSWER, "Visual7W":POS_MOD_ANSWER,"ScienceQA":POS_MOD_ANSWER, "GQA":POS_MOD_ANSWER, "TextVQA":POS_MOD_ANSWER, "VizWiz":POS_MOD_ANSWER,
                
                "MSCOCO_i2t":POS_MOD_IMAGE_CAPTION, "VisualNews_i2t":POS_MOD_IMAGE_CAPTION,
                }

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None

    width, height = image.size
    max_side = max(width, height)

    if resolution == "high":
        target_max = 1344
    elif resolution == "mid":
        target_max = 672
    elif resolution == "low":
        target_max = 448
    elif resolution == "tiny":
        target_max = 336
    else:
        target_max = max_dim

    # Tính tỉ lệ scale sao cho cạnh lớn nhất = target_max
    if max_side > target_max:
        scale = target_max / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height))

    return image

def create_semi_orthogonal_matrix(tensor):
    rows, cols = tensor.shape
    if rows >= cols:
        # QR trực tiếp
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q[:, :cols]
    else:
        # QR trên ma trận transpose để đảm bảo W W^T = I
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode='reduced')
        tensor.data[:] = q.T[:rows, :]
    return tensor

class SingleWrapper(nn.Module):
    def __init__(self, model_args, training_args):
        super(SingleWrapper, self).__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model = self._load_model() # MMEBModel
        self.temperature = model_args.temperature
        self.student_hidden_dim = self.model_args.student_hidden_dim
        self.teacher_hidden_dim = self.model_args.teacher_hidden_dim
        self.set_projector()
    
    def _load_model(self):
        print("Load single model with lora rank:", self.model_args.lora_r)
        print("Model use lora:", self.model_args.lora)
        model = MMEBModel.build(self.model_args)
        print("Model built.")
        return model 
    
    def get_processor(self):
        if hasattr(self, 'processor'):
            return self.processor
        processor = load_processor(self.model_args, None)
        setattr(self, 'processor', processor)
        print("Processor loaded.")
        return processor

    def set_projector(self):
        """
        Create a list of linear projectors mapping
        student_hidden_dim -> teacher_hidden_dim
        One projector per teacher layer mapping.
        """
        projector_list = nn.ModuleList()
        
        if self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            projector_config = json.load(open(self.model_args.projector_config_path, 'r'))
            
            name_dict = {
                "s": self.student_hidden_dim,
                "t": self.teacher_hidden_dim,
                "relu": nn.ReLU()
            }
            
            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue
                seq = nn.Sequential()
                parts = cfg["structure"].split("-")
                parsed = []
                
                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    else:
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * name_dict[p[-1]])
                for i in range(len(parsed) -1):
                    a, b = parsed[i], parsed[i+1]
                    if isinstance(a, int) and isinstance(b, int):
                        layer = nn.Linear(a, b)
                        # create_semi_orthogonal_matrix(layer.weight)
                        with torch.no_grad():
                            layer.weight.normal_(mean=0.0, std=1e-3)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                    elif b == "relu":
                        seq.append(name_dict[b])
                    elif a =="relu" and isinstance(b, int):
                        prev_out = parsed[i-1] if isinstance(parsed[i-1], int) else None
                        layer = nn.Linear(prev_out, b)
                        # create_semi_orthogonal_matrix(layer.weight)
                        with torch.no_grad():
                            layer.weight.normal_(mean=0.0, std=1e-3)
                        layer = layer.to(dtype=torch.bfloat16)
                        seq.append(layer)
                self.projectors[name] = seq
        elif self.training_args.teacher_layer_mapping:
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector = nn.Linear(
                    self.student_hidden_dim,
                    self.teacher_hidden_dim,
                    dtype=torch.bfloat16
                )
                with torch.no_grad():
                    projector.weight.normal_(mean=0.0, std=1e-3)
                projector_list.append(projector)

            self.projectors = projector_list
        elif self.training_args.num_projectors > 0:
            for _ in range(self.training_args.num_projectors):
                projector = nn.Linear(
                    self.student_hidden_dim,
                    self.teacher_hidden_dim,
                    dtype=torch.bfloat16
                )
                with torch.no_grad():
                    projector.weight.normal_(mean=0.0, std=1e-3)
                projector_list.append(projector)

            self.projectors = nn.ModuleList(projector_list)
        else:
            self.projectors = None
            print("No projectors created.")

        if self.projectors:
            print(f"Created {len(self.projectors)} linear projectors.")
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, 'projectors') and self.projectors is not None:
            lr = getattr(self.training_args, "projector_lr", None) or self.training_args.learning_rate
            optimizer.add_param_group({
                "params": self.projectors.parameters(),
                "lr": lr
            })
            print("-----------Projector parameters added to optimizer.----------")
        
        return optimizer
    
    def forward(self, criterion, batch):
        loss = criterion(self, batch)
        return loss

class SingleCollator:
    def __init__(self, processor: ProcessorMixin, model_args: ModelArguments, 
                 data_args: DataArguments, training_args: TrainingArguments,
                 batch_size: Optional[int] = None):
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size
    
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        # print("Processing batch for keys:", text_keyname, image_keyname)
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = ' ', None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                if not isinstance(text, list):
                    text = [text]
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                if not text and not raw_images:
                    text, visual_input = ' ', None
                    texts.append(text)
                    visual_inputs.append(visual_input)
                else:
                    for t, img in zip(text, raw_images):
                        if not t and img is None:
                            t, img = ' ', None
                        texts.append(t)
                        visual_inputs.append(img)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs
    
    def __call__(self, examples):
        qry_inputs = self._get_batch_inputs(examples, "query_text", "query_image")
        pos_inputs = self._get_batch_inputs(examples, "pos_text", "pos_image")

        bs = len(qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'
        
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")
        
        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]

        processed_qry_inputs = process_fn(qry_inputs, processor=self.processor, 
                                                          max_length=self.data_args.max_len, 
                                                          )
        processed_pos_inputs = process_fn(pos_inputs, processor=self.processor, 
                                                          max_length=self.data_args.max_len,
                                                          )
        
        # get encoded_dir from examples
        if self.data_args.caching_dir:
            teacher_qry_cache = [ex["teacher_qry_cache"] for ex in examples] # list of dicts
            teacher_pos_cache = [ex["teacher_pos_cache"] for ex in examples] 
        else:
            teacher_qry_cache, teacher_pos_cache = None, None

        return {
            'qry': processed_qry_inputs,
            'pos': processed_pos_inputs,
            "teacher_qry_caches": teacher_qry_cache,
            "teacher_pos_caches": teacher_pos_cache,
            "encoded_dir": [ex["encoded_dir"] for ex in examples]
        }

class SingleDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, 
                subset,
                split=f"{self.data_args.dataset_split}"
            )
            if subset == "WebQA" and "qry" in subset_data.column_names:
                subset_data = subset_data.map(
                    lambda x: {"qry": x["qry"].replace("<|image_1|>", "").strip()}
                )
                print_rank("Preprocessed WebQA to remove <image_1> tokens in queries.")
            total_samples = len(subset_data)
            num_samples_to_keep = math.ceil(total_samples * self.data_args.percent_data)
            subset_data = subset_data.select(range(num_samples_to_keep))
            subset_data = subset_data.add_column("pos_text_instruction", [POS_MOD_DICT.get(subset, "") + text for text in subset_data['pos_text']])
            subset_data = subset_data.remove_columns(set(['neg_text', 'neg_image_path']) & set(subset_data.column_names))
            subset_data = subset_data.remove_columns(set(subset_data.column_names) - set(['qry', 'qry_image_path', 'pos_image_path', 'pos_text_instruction']))
            subset_data = subset_data.rename_column("pos_text_instruction", "pos_text")
            subset_data = subset_data.add_column(
                "encoded_dir",
                [f"{subset}/{idx}" for idx in range(len(subset_data))]
            )
            train_data.append(subset_data)
            
        self.train_data = concatenate_datasets(train_data)
        print_rank(f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}")
    
    def __len__(self):
        return len(self.train_data)
    
    def _get_image(self, img_path, backbone):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        image = image.convert("RGB")
        width, height = image.size
        MIN_SIZE = 16
        if width < MIN_SIZE or height < MIN_SIZE:
            new_width = max(width, MIN_SIZE)
            new_height = max(height, MIN_SIZE)
            result = Image.new(image.mode, (new_width, new_height), (0,0,0))
            x_offset = (new_width - width) // 2
            y_offset = (new_height - height) // 2
            result.paste(image, (x_offset, y_offset))
            image = result
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
    
    def _get_cache(self, data_idx, name="qry.pt"):
        if not self.data_args.caching_dir:
            return None
        cache = torch.load(os.path.join(self.data_args.caching_dir, self.train_data[data_idx]["encoded_dir"], name), map_location="cpu")
        if isinstance(cache, dict):
            return cache
        return {
            "rep": cache,
            "mean_last_img_token": None,
            "mean_last_text_token": None,
        }

    def __getitem__(self, data_idx):
        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)
        
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )

        # Đảm bảo dữ liệu ở dạng list để xử lý đồng nhất
        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            
        final_qry_texts, final_qry_images, final_pos_texts, final_pos_images = [], [], [], []
        
        # Chỉ lấy backbone của model chính
        model_backbone = self.model_args.model_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2, qwenvl
            
            curr_qry_text, curr_pos_text = qry_text, pos_text
            
            # Thay thế token ảnh nếu backbone không phải là PHI3V
            if model_backbone != PHI3V:
                curr_qry_text = curr_qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
                curr_pos_text = curr_pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
            
            # Load ảnh
            curr_qry_image = self._get_image(qry_image_path, model_backbone)
            curr_pos_image = self._get_image(pos_image_path, model_backbone)

            # Kiểm tra input rỗng
            if (not curr_qry_text and not curr_qry_image) or (not curr_pos_text and not curr_pos_image):
                print("empty inputs")
                continue
            
            final_qry_texts.append(curr_qry_text)
            final_qry_images.append(curr_qry_image)
            final_pos_texts.append(curr_pos_text)
            final_pos_images.append(curr_pos_image)

        tea_qry_cache = self._get_cache(data_idx, name="qry.pt")
        tea_pos_cache = self._get_cache(data_idx, name="pos.pt")

        return {
            "query_text": final_qry_texts,
            "query_image": final_qry_images,
            "pos_text": final_pos_texts,
            "pos_image": final_pos_images,
            "teacher_qry_cache": tea_qry_cache,
            "teacher_pos_cache": tea_pos_cache,
            "encoded_dir": self.train_data[data_idx]["encoded_dir"]
        }