#!/bin/bash

# Example of training usage

# Choose a training task:
# 1. Default PP-OCR recognition model
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7' tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml

# 2. Meter reading detection model
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/det/meter/meter_det_mv3.yml

# 3. Meter reading recognition model (standard)
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/rec/meter/meter_PP-OCRv3_rec.yml

# 4. Meter reading recognition model (distillation)
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/rec/meter/meter_PP-OCRv3_rec_distillation.yml

# For easier training of meter reading models, use the dedicated scripts:
# - Detection: ./model_training/det_train/scripts/train_meter_detection.sh
# - Recognition: ./model_training/rec_train/scripts/train_meter_recognition.sh

# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
