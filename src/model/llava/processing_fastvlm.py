"""
Processor class for FastVLM.
"""

import math 
from collections.abc import Iterable
from typing import Optional, Union, Optional

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from transformers import CLIPImageProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from src.model.llava.mm_utils import tokenizer_image_token
from src.model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

logger = logging.get_logger(__name__)

def expand2square(pil_img, background_color):
    pil_img = pil_img.convert("RGB")
    width, height = pil_img.size
    MIN_SIZE = 32
    if width < MIN_SIZE or height < MIN_SIZE:
        new_width = max(width, MIN_SIZE)
        new_height = max(height, MIN_SIZE)
        
        result = Image.new(pil_img.mode, (new_width, new_height), background_color)
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        result.paste(pil_img, (x_offset, y_offset))
        pil_img = result
        
        width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class FastVLMProcessor(ProcessorMixin):
    r"""
    Constructs a FastVLM processor which wraps a FastVLM image processor and a tokenizer into a single processor.
    [`FastVLMProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`PreTrainedTokenizer`]. See the
    [`~FastVLMProcessor.__call__`] and [`~FastVLMProcessor.decode`] for more information.
    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = []
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        super().__init__(image_processor, tokenizer, **kwargs)
        
    def __call__(
        self, 
        images: Optional[ImageInput] = None,
        texts: Optional[Union[TextInput, PreTokenizedInput, Iterable[TextInput], Iterable[PreTokenizedInput]]] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        input_ids = [tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for text in texts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        image_tensors = []
        for image in images:
            if image is not None: 
                image = image.convert("RGB")
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensors.append(image)
        
        if len(image_tensors) > 0: 
            image_tensors = torch.stack(image_tensors, dim=0)
        else:
            image_tensors = None
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if image_tensors is not None:
            data["images"] = image_tensors

        return BatchFeature(data=data, tensor_type="pt")
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
    def post_process_image_text_to_text(self, generated_outputs):
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

class FastVLMProcessor2(ProcessorMixin):
    r"""
    Constructs a FastVLM processor which wraps a FastVLM image processor and a tokenizer into a single processor.
    [`FastVLMProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`PreTrainedTokenizer`]. See the
    [`~FastVLMProcessor.__call__`] and [`~FastVLMProcessor.decode`] for more information.
    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = []
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        super().__init__(image_processor, tokenizer, **kwargs)
        self.patch_size = 64
        
    def __call__(
        self, 
        images: Optional[ImageInput] = None,
        texts: Optional[Union[TextInput, PreTokenizedInput, Iterable[TextInput], Iterable[PreTokenizedInput]]] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        input_ids = [tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for text in texts]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        target_longest_edge = 1024
        patch_size = self.patch_size
        image_tensors = []

        for image in images:
            if image is not None: 
                image = image.convert("RGB")
                W, H = image.size

                # 1. Tính scale dựa trên cạnh dài nhất
                scale = target_longest_edge / max(W, H)
                
                # 2. Tính kích thước mới thô (raw)
                raw_new_w = W * scale
                raw_new_h = H * scale

                # 3. "Snap" (Làm tròn) kích thước về bội số của patch_size (64)
                # Dùng hàm round() để lấy bội số gần nhất nhằm giữ tỉ lệ chuẩn nhất có thể
                new_w = int(round(raw_new_w / patch_size) * patch_size)
                new_h = int(round(raw_new_h / patch_size) * patch_size)

                # Đảm bảo kích thước tối thiểu là 1 patch (tránh lỗi nếu ảnh quá dẹt)
                new_w = max(new_w, patch_size)
                new_h = max(new_h, patch_size)
                
                # 4. Resize trực tiếp (Không padding)
                # Ảnh sẽ bị méo cực nhẹ để khớp vào lưới patch_size
                image = image.resize((new_w, new_h), resample=Image.BICUBIC)

                # 5. Preprocess (Chỉ normalize, tắt resize/crop của processor)
                pixel_values = self.image_processor.preprocess(
                    image, 
                    do_resize=False, 
                    do_center_crop=False, 
                    return_tensors='pt'
                )['pixel_values'][0]

                image_tensors.append(pixel_values)
        
        if len(image_tensors) == 0: 
            image_tensors = None
        else:
            shapes = [img.shape for img in image_tensors]
            # tất cả shape giống nhau?
            if all(s == shapes[0] for s in shapes):
                image_tensors = torch.stack(image_tensors, dim=0)

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if image_tensors is not None:
            data["images"] = image_tensors

        return BatchFeature(data=data)
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    
    def post_process_image_text_to_text(self, generated_outputs):
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

__all__ = ["FastVLMProcessor", "FastVLMProcessor2"]