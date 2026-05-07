#!/bin/bash
# ==============================================================================
# Script huấn luyện mô hình KEHN Joint Learning trên Google Colab
# Hướng dẫn sử dụng:
# 1. Mở Google Colab, tạo Notebook mới và bật GPU (T4/A100).
# 2. Tạo một ô Code và chạy lệnh sau để Mount Google Drive:
#    from google.colab import drive
#    drive.mount('/content/drive')
# 3. Tạo một ô Code tiếp theo để Clone Repo và chạy script:
#    !git clone https://github.com/KwanFam26022005/Medical-NLU-Pipeline.git
#    %cd Medical-NLU-Pipeline
#    !bash train_kehn_colab.sh
# ==============================================================================

# Cài đặt các thư viện cần thiết
echo "Cài đặt các thư viện cần thiết..."
pip install -q -r requirements.txt

# Cấu hình đường dẫn lưu trữ mô hình (Lưu thẳng vào Google Drive để không bị mất khi Colab disconnect)
export MODEL_DIR="/content/drive/MyDrive/Medical-NLU-Pipeline/outputs/kehn_joint_model"
mkdir -p $MODEL_DIR

echo "Bắt đầu huấn luyện mô hình KEHN Joint Learning..."
# Chạy script huấn luyện với module python (sử dụng -m để nhận diện package)
python -m kehn_joint.train_joint \
    --exp_name E4_kehn_vihealthbert \
    --backbone vihealthbert \
    --epochs 30 \
    --batch_size 16 \
    --lr 3e-5

echo "Huấn luyện hoàn tất! Tiến hành đánh giá Benchmark..."
python -m kehn_joint.evaluate_joint

echo "Tiến hành sao chép kết quả sang Google Drive..."
# Copy kết quả sang Drive (vì theo config_joint.py, output được lưu ở thư mục kehn_joint/outputs)
cp -r kehn_joint/outputs/E4_kehn_vihealthbert/* $MODEL_DIR/
cp kehn_joint/outputs/benchmark_report.md $MODEL_DIR/ || true

echo "Đã sao chép mô hình và báo cáo Benchmark sang Google Drive ($MODEL_DIR)."
