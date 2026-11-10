"""
train_full.py - Huấn luyện mô hình phát hiện hung khí (bộ 8 lớp vũ khí thật).

Đã cân bằng dữ liệu và bổ sung hard negatives từ SOHAS:
- 8 lớp vũ khí: Axe, Sickle, Sword, Crowbar, Knife, Pistol, Long_gun, Grenade
- Bỏ 4 lớp gây nhiễu: Chain, Stick, Hammer, Spanner
- 20.9% hard negatives (cắt triệt để lỗi nhận diện cánh tay, balo thành hung khí)

Cách chạy:
    python train_full.py                      # Chạy full 80 epoch từ yolo26s.pt
    python src/training/train.py --mode full  # Tương đương lệnh trên
    python src/training/train.py --mode finetune # Tinh chỉnh nhanh 15 epoch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.training.train import main

if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "full"])
    if "--data" not in sys.argv:
        sys.argv.extend(["--data", "data/weapon_8cls.yaml"])
    if "--name" not in sys.argv:
        sys.argv.extend(["--name", "weapon_full_8cls"])
    main()
