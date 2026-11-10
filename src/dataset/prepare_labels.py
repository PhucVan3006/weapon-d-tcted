"""
Dataset Label Normalization & Noise Cleaning Module.
Implements Stage 1 of Training Pipeline:
- Merges 4 original classes (Knife: 0, Pistol: 1, Rifle: 2, Stick-Rod: 3)
  into 3 standard target classes:
    0: gun (merges Pistol & Rifle)
    1: knife (Knife)
    2: bat_stick (Stick-Rod)
- Removes noisy/garbage bounding boxes (< 10x10 pixels or extreme aspect ratio).

-----------------------------------------------------------------------------
SỬA LỖI v2 - hai thay đổi
-----------------------------------------------------------------------------
1. CHẶN CHẠY LẠI LẦN HAI.
   Script sửa nhãn TẠI CHỖ theo ánh xạ 0->1, 1->0, 2->0, 3->2. Chạy lần
   hai trên nhãn ĐÃ remap sẽ đảo gun<->knife và biến bat_stick thành gun
   - hỏng dataset mà không báo lỗi gì. Nay nếu thư mục labels_backup_raw
   đã tồn tại, script dừng lại và yêu cầu --force.

2. CHỈ LÀM SẠCH TẬP TRAIN.
   Bản cũ xoá hộp nhỏ ở cả valid và test. Hai hậu quả:
     - Vật thể nhỏ (dao/súng ở xa trong khung CCTV - đúng trường hợp khó
       nhất và quan trọng nhất) biến mất khỏi tập đánh giá.
     - Recall báo cáo bị THỔI PHỒNG, hệ thống trông tốt hơn thực tế.
   Tập đánh giá phải phản ánh thực tế, không được "dọn dẹp". Nay mặc định
   chỉ xử lý train; muốn đụng vào valid/test phải chỉ định rõ --splits.
"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path

CLASS_MAPPING = {
    0: 1,  # Knife -> knife (1)
    1: 0,  # Pistol -> gun (0)
    2: 0,  # Rifle -> gun (0)
    3: 2   # Stick-Rod -> bat_stick (2)
}

TARGET_NAMES = {0: "gun", 1: "knife", 2: "bat_stick"}


def already_processed(workspace_root: str, splits: list) -> list:
    """Trả về danh sách split đã từng được remap (có labels_backup_raw)."""
    done = []
    for split in splits:
        if os.path.isdir(os.path.join(workspace_root, split, "labels_backup_raw")):
            done.append(split)
    return done


def clean_and_remap_labels(
    workspace_root: str = "c:/ptchttm",
    splits: list = None,
    min_box_size_px: float = 10.0,
    img_size: float = 640.0,
    backup: bool = True,
    force: bool = False
):
    # Mặc định CHỈ train. Xem ghi chú 2 ở đầu file.
    if splits is None:
        splits = ["train"]

    print("=" * 60)
    print("STAGE 1: NORMALIZING LABELS & REMOVING NOISE BOUNDING BOXES")
    print(f"Workspace: {workspace_root}")
    print(f"Splits   : {', '.join(splits)}")
    print(f"Minimum box size: {min_box_size_px}x{min_box_size_px} pixels (on {img_size}x{img_size} scale)")
    print("=" * 60)

    # --- Chặn chạy lần hai (xem ghi chú 1 ở đầu file) -------------------
    done = already_processed(workspace_root, splits)
    if done and not force:
        print("\n" + "!" * 60)
        print("  DỪNG LẠI - CÁC SPLIT SAU ĐÃ ĐƯỢC REMAP TRƯỚC ĐÓ:")
        for s in done:
            print(f"    - {s}  (đã có {s}/labels_backup_raw)")
        print()
        print("  Chạy lại sẽ ĐẢO NGƯỢC nhãn: gun <-> knife, bat_stick -> gun.")
        print("  Dataset sẽ hỏng mà không có thông báo lỗi nào.")
        print()
        print("  Nếu bạn CHẮC CHẮN muốn chạy lại, hãy khôi phục nhãn gốc từ")
        print("  labels_backup_raw trước, rồi chạy kèm --force.")
        print("!" * 60)
        return False

    min_norm_size = min_box_size_px / img_size
    min_area = min_norm_size * min_norm_size

    for split in splits:
        label_dir = os.path.join(workspace_root, split, "labels")
        if not os.path.exists(label_dir):
            print(f"[-] Directory {label_dir} does not exist, skipping...")
            continue

        backup_dir = os.path.join(workspace_root, split, "labels_backup_raw")
        if backup and not os.path.exists(backup_dir):
            print(f"[+] Creating backup of original labels in: {backup_dir}")
            os.makedirs(backup_dir, exist_ok=True)
            for f in os.listdir(label_dir):
                if f.endswith(".txt"):
                    with open(os.path.join(label_dir, f), "r", encoding="utf-8") as src, \
                         open(os.path.join(backup_dir, f), "w", encoding="utf-8") as dst:
                        dst.write(src.read())

        total_files = 0
        cleaned_files = 0
        total_boxes_before = 0
        total_boxes_after = 0
        removed_tiny_boxes = 0
        new_class_counts = Counter()

        for filename in os.listdir(label_dir):
            if not filename.endswith(".txt"):
                continue

            total_files += 1
            file_path = os.path.join(label_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                total_boxes_before += 1
                orig_cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                if (w * img_size < min_box_size_px and h * img_size < min_box_size_px) or (w * h < min_area):
                    removed_tiny_boxes += 1
                    continue

                if orig_cls in CLASS_MAPPING:
                    new_cls = CLASS_MAPPING[orig_cls]
                else:
                    new_cls = orig_cls

                new_class_counts[new_cls] += 1
                total_boxes_after += 1
                new_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(file_path, "w", encoding="utf-8") as f:
                if new_lines:
                    f.write("\n".join(new_lines) + "\n")
                else:
                    f.write("")

            cleaned_files += 1

        print(f"\n[OK] Split: {split.upper()}")
        print(f"  Processed files   : {cleaned_files}/{total_files}")
        print(f"  Boxes before      : {total_boxes_before}")
        print(f"  Boxes after       : {total_boxes_after}")
        print(f"  Removed tiny noise: {removed_tiny_boxes}")
        print(f"  Class breakdown   :")
        for cls_id in sorted(new_class_counts.keys()):
            name = TARGET_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    Class {cls_id} ({name}): {new_class_counts[cls_id]} boxes")

    print("\n[SUCCESS] Label normalization and cleaning complete!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize dataset labels to 3 classes and filter noise")
    parser.add_argument("--root", type=str, default="c:/ptchttm", help="Path to workspace root")
    parser.add_argument("--min-px", type=float, default=10.0, help="Min box size in pixels to keep")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backup directory")
    parser.add_argument("--splits", type=str, default="train",
                        help="Danh sách split, ngăn cách bằng dấu phẩy. Mặc định chỉ 'train'. "
                             "KHÔNG nên thêm valid/test: xoá hộp nhỏ khỏi tập đánh giá "
                             "sẽ làm recall báo cáo cao hơn thực tế.")
    parser.add_argument("--force", action="store_true",
                        help="Bỏ qua chốt chặn chạy lần hai. Chỉ dùng sau khi đã "
                             "khôi phục nhãn gốc từ labels_backup_raw.")
    args = parser.parse_args()
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    clean_and_remap_labels(
        workspace_root=args.root,
        splits=split_list,
        min_box_size_px=args.min_px,
        backup=not args.no_backup,
        force=args.force
    )
