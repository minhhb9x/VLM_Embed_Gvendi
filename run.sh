# !/bin/bash

# FastVLM-0.5B 
bash scripts/train_distill_gvendi_cls.sh 0 1 1.0 0.1 1 12347
bash scripts/train_distill_gvendi_cls.sh 0 1 1.0 0.5 1 12346
bash scripts/train_distill_gvendi_cls.sh 0 1 1.0 1.0 1 12348

bash scripts/train_distill_gvendi_cls.sh 1 1 1.0 0.5 1 12340
bash scripts/train_distill_gvendi_cls.sh 1 0 1.0 0.5 1 12341