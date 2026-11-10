"""
process_sohas.py - Trộn dataset SOHAS vào dataset của dự án.

ĐÃ SỬA MỘT LỖI QUAN TRỌNG
-------------------------
Bản trước lặp qua [('train','train'), ('test','test')] nên split TEST của
SOHAS bị đổ thẳng vào tập TEST của dự án. Hai hậu quả:
  - Tập test bị đổi -> mốc mAP50 0.8504 không còn so sánh được.
  - Split test của SOHAS lại chính là phần ĐÚNG MIỀN NHẤT (frame trích từ
    video thật, trung vị hộp 2-5% khung so với 43% ở split train). Để nó ở
    test thì mô hình không bao giờ học được từ nó.
Nay CẢ HAI split của SOHAS đều vào train/valid, chia 85/15 theo băm MD5 của
tên file. Tập test của dự án KHÔNG bị chạm tới, giữ nguyên làm mốc đo.

Nếu đã chạy bản cũ với --apply rồi thì chạy fix_sohas_split.py để sửa.

Follows Section 1.5 of CLONE_DATASET.txt:
  B1. Filter out pistol in TRAIN with box area > 0.25 (keep small pistols).
  B2. Images with ONLY handheld objects {1, 3, 4, 5} become HARD NEGATIVES (empty .txt).
  B3. Remap remaining weapon boxes:
        SOHAS 0 (pistol) -> Project 5 (Pistol)
        SOHAS 2 (knife)  -> Project 4 (Knife)
      Drop handheld lines {1, 3, 4, 5} in mixed images.
  B4. Ensure hard negatives ratio <= 30% of total train.
  B5. Audit dataset metrics.
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from collections import Counter

SOHAS_DIR = Path(r"c:\ptchttm\sohas\Weapons and similar handled objects\Sohas_weapon-Detection-YOLOv5\obj_train_data")
TARGET_DIR = Path(r"c:\ptchttm\data\dataset_merged")

PROJECT_CLASS_MAP = {
    0: 5,  # SOHAS pistol -> Project Pistol (5)
    2: 4,  # SOHAS knife  -> Project Knife (4)
}

HANDHELD_IDS = {1, 3, 4, 5}  # smartphone, monedero, billete, tarjeta

VALID_PCT = 15  # phần trăm đưa sang valid; phần còn lại vào train


def bucket(stem: str) -> int:
    """0-99, cố định theo tên file nên chia lại lần nào cũng như nhau."""
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16) % 100


def dest_split(stem: str) -> str:
    """Đích của một ảnh. KHÔNG BAO GIỜ trả về 'test' - xem ghi chú đầu file."""
    return "valid" if bucket(stem) < VALID_PCT else "train"

def analyze_sohas(apply=False):
    print("=" * 70)
    print("PROCESSING SOHAS DATASET (CLONE_DATASET.txt Section 1.5)")
    print("=" * 70)
    
    for split_source in ('train', 'test'):
        img_src = SOHAS_DIR / "images" / split_source
        lbl_src = SOHAS_DIR / "labels" / split_source

        # Đích được quyết định TỪNG ẢNH ở dest_split(), chỉ train hoặc valid.
        if apply:
            for sp in ("train", "valid"):
                (TARGET_DIR / sp / "images").mkdir(parents=True, exist_ok=True)
                (TARGET_DIR / sp / "labels").mkdir(parents=True, exist_ok=True)
            
        stats = {
            "total_images": 0,
            "pistol_dropped_large": 0,
            "pistol_kept_small": 0,
            "knife_kept": 0,
            "hard_negative_created": 0,
            "weapon_images_added": 0,
            "completely_discarded": 0,
            "to_train": 0,
            "to_valid": 0,
        }
        
        img_files = [f for f in img_src.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        stats["total_images"] = len(img_files)
        
        for img_path in img_files:
            lbl_path = lbl_src / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            raw_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    raw_boxes.append((cid, coords))
                    
            # Check for handheld only
            cids = [b[0] for b in raw_boxes]
            has_handheld_only = (len(cids) > 0 and all(c in HANDHELD_IDS for c in cids))
            
            # Filter weapon boxes
            kept_weapon_boxes = []
            for cid, coords in raw_boxes:
                if cid == 0:  # Pistol
                    area = coords[2] * coords[3]
                    if split_source == 'train' and area > 0.25:
                        stats["pistol_dropped_large"] += 1
                        continue  # B1: Drop large pistol in train
                    stats["pistol_kept_small"] += 1
                    target_cid = PROJECT_CLASS_MAP[0]
                    kept_weapon_boxes.append((target_cid, coords))
                elif cid == 2:  # Knife
                    stats["knife_kept"] += 1
                    target_cid = PROJECT_CLASS_MAP[2]
                    kept_weapon_boxes.append((target_cid, coords))
                else:
                    # Handheld class {1, 3, 4, 5}
                    pass
                    
            new_img_name = f"sohas_{img_path.name}"
            new_lbl_name = f"sohas_{img_path.stem}.txt"
            sp = dest_split(f"sohas_{img_path.stem}")
            img_dst = TARGET_DIR / sp / "images"
            lbl_dst = TARGET_DIR / sp / "labels"
            
            if kept_weapon_boxes:
                # Normal weapon image
                stats["weapon_images_added"] += 1
                stats["to_train" if sp == "train" else "to_valid"] += 1
                if apply:
                    shutil.copy2(img_path, img_dst / new_img_name)
                    with open(lbl_dst / new_lbl_name, "w", encoding="utf-8") as out_f:
                        for target_cid, coords in kept_weapon_boxes:
                            out_f.write(f"{target_cid} " + " ".join(f"{x:.6f}" for x in coords) + "\n")
            elif has_handheld_only:
                # B2: Hard negative image (empty txt)
                stats["hard_negative_created"] += 1
                stats["to_train" if sp == "train" else "to_valid"] += 1
                if apply:
                    shutil.copy2(img_path, img_dst / new_img_name)
                    # Write empty label file
                    open(lbl_dst / new_lbl_name, "w", encoding="utf-8").close()
            else:
                # Only contained dropped large pistols
                stats["completely_discarded"] += 1
                
        print(f"\n--- SOHAS split {split_source.upper()} -> train/valid ---")
        print(f"  Total SOHAS images evaluated : {stats['total_images']}")
        print(f"  Pistol boxes dropped (>0.25)  : {stats['pistol_dropped_large']}")
        print(f"  Pistol boxes kept (<=0.25)    : {stats['pistol_kept_small']}")
        print(f"  Knife boxes kept              : {stats['knife_kept']}")
        print(f"  Hard Negative images created  : {stats['hard_negative_created']}")
        print(f"  Weapon images added           : {stats['weapon_images_added']}")
        print(f"  Images completely discarded   : {stats['completely_discarded']}")
        print(f"  -> train                      : {stats['to_train']}")
        print(f"  -> valid                      : {stats['to_valid']}")
        print(f"  -> test                       : 0  (cố ý: giữ mốc đo)")

if __name__ == "__main__":
    is_apply = "--apply" in sys.argv
    analyze_sohas(apply=is_apply)
    if not is_apply:
        print("\n[!] Dry-run complete. Run with --apply to execute copy and merge.")

