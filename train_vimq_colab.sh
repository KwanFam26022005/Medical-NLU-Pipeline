#!/bin/bash
# ==============================================================================
# Script huấn luyện mô hình Joint (ViMQ) trên Google Colab
# Hướng dẫn sử dụng:
# 1. Mở Google Colab, tạo Notebook mới và bật GPU (T4).
# 2. Tạo một ô Code và chạy lệnh sau để Mount Google Drive:
#    from google.colab import drive
#    drive.mount('/content/drive')
# 3. Tạo một ô Code tiếp theo để Clone Repo và chạy script:
#    !git clone https://github.com/KwanFam26022005/Medical-NLU-Pipeline.git
#    %cd Medical-NLU-Pipeline
#    !bash train_vimq_colab.sh
# ==============================================================================

# Cài đặt các thư viện cần thiết
echo "Cài đặt các thư viện cần thiết..."
pip install -q transformers seqeval torch pyvi

# Cấu hình đường dẫn lưu trữ mô hình (Lưu thẳng vào Google Drive để không bị mất khi Colab disconnect)
export MODEL_DIR="/content/drive/MyDrive/Medical-NLU-Pipeline/outputs/vimq_joint_model"
mkdir -p $MODEL_DIR

# Di chuyển vào thư mục src của ViMQ để chạy
cd ViMQ-main/ViMQ-main/src/

echo "Bắt đầu huấn luyện mô hình ViMQ Joint Learning..."
# Chạy script huấn luyện với các siêu tham số (Hyperparameters) tối ưu
python main.py \
    --model_type vimq_model \
    --model_dir $MODEL_DIR \
    --data_dir ../data \
    --seed 42 \
    --do_train \
    --do_eval \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --num_iteration 2 \
    --save_steps 500 \
    --logging_steps 100 \
    --tuning_metric f1_score \
    --gpu_id 0 \
    --max_seq_len 256 \
    --iternoise 1 \
    --omega 0 \
    --threshold_iou 0.9 \
    --lamda 3

echo "Huấn luyện hoàn tất! Mô hình được lưu tại: $MODEL_DIR"
