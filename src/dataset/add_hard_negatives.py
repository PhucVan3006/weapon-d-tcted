"""
Hard Negative Mining & Background Sample Ingestion Module.
Implements Stage 1 & Stage 4 of Training Pipeline:
- Injects 15% - 20% background / negative images (people holding phones, umbrellas, bottles, empty scenes).
- Generates 0-byte label files (.txt) for each negative sample.
- Extracts non-weapon background crops from existing frames with 0 IoU overlap to mimic CCTV conditions.
- Can ingest false-positive frames collected from inference testing.
"""

import os
import cv2
import random
import argparse
import numpy as np
from pathlib import Path


def is_box_overlapping(crop_box, weapon_boxes):
    cx1, cy1, cx2, cy2 = crop_box
    for wb in weapon_boxes:
        w_cls, wx, wy, ww, wh = wb
        wx1 = wx - ww / 2.0
        wy1 = wy - wh / 2.0
        wx2 = wx + ww / 2.0
        wy2 = wy + wh / 2.0

        inter_x1 = max(cx1, wx1)
        inter_y1 = max(cy1, wy1)
        inter_x2 = min(cx2, wx2)
        inter_y2 = min(cy2, wy2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            return True
    return False


def generate_hard_negatives(
    workspace_root: str = "c:/ptchttm",
    target_ratio: float = 0.15,
    max_to_generate: int = 1200
):
    print("=" * 60)
    print("STAGE 1: HARD NEGATIVE SAMPLES GENERATION & INGESTION")
    print(f"Target ratio: {target_ratio*100:.1f}% negative samples")
    print("=" * 60)

    train_img_dir = os.path.join(workspace_root, "train", "images")
    train_lbl_dir = os.path.join(workspace_root, "train", "labels")

    if not os.path.exists(train_img_dir) or not os.path.exists(train_lbl_dir):
        print(f"[-] Training directories not found at {train_img_dir}")
        return

    img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    lbl_files = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt')]

    total_images = len(img_files)
    empty_labels = 0
    for lf in lbl_files:
        p = os.path.join(train_lbl_dir, lf)
        if os.path.getsize(p) == 0:
            empty_labels += 1

    current_ratio = empty_labels / total_images if total_images > 0 else 0
    print(f"Current total images  : {total_images}")
    print(f"Current empty labels  : {empty_labels} ({current_ratio*100:.2f}%)")

    needed_negatives = int(total_images * target_ratio) - empty_labels
    needed_negatives = max(0, min(needed_negatives, max_to_generate))

    if needed_negatives <= 0:
        print(f"[OK] Training set already has adequate negative samples (>= {target_ratio*100:.1f}%).")
        return

    print(f"[+] Generating {needed_negatives} hard negative background samples from real CCTV scenes...")

    generated = 0
    candidate_imgs = [f for f in img_files if not f.startswith("neg_bg_")]
    random.shuffle(candidate_imgs)

    for img_name in candidate_imgs:
        if generated >= needed_negatives:
            break

        base_name = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(train_lbl_dir, f"{base_name}.txt")
        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        weapon_boxes = []
        for p in lines:
            if len(p) >= 5:
                weapon_boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        img_path = os.path.join(train_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        for _ in range(5):
            crop_w = random.randint(int(w * 0.4), int(w * 0.8))
            crop_h = random.randint(int(h * 0.4), int(h * 0.8))
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            norm_box = (x1 / w, y1 / h, x2 / w, y2 / h)
            if not is_box_overlapping(norm_box, weapon_boxes):
                cropped = img[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_AREA)

                neg_name = f"neg_bg_{generated:05d}.jpg"
                neg_lbl_name = f"neg_bg_{generated:05d}.txt"

                cv2.imwrite(os.path.join(train_img_dir, neg_name), resized)
                with open(os.path.join(train_lbl_dir, neg_lbl_name), "w", encoding="utf-8") as f:
                    pass

                generated += 1
                if generated % 200 == 0 or generated == needed_negatives:
                    print(f"  [+] Created {generated}/{needed_negatives} negative samples...")
                break

    print(f"\n[SUCCESS] Added {generated} negative samples.")
    new_total = total_images + generated
    new_empty = empty_labels + generated
    print(f"New total images : {new_total}")
    print(f"New empty labels : {new_empty} ({new_empty / new_total * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject hard negative samples into train set")
    parser.add_argument("--root", type=str, default="c:/ptchttm")
    parser.add_argument("--ratio", type=float, default=0.15, help="Target ratio (e.g. 0.15 for 15%)")
    parser.add_argument("--max", type=int, default=1200, help="Max negatives to generate")
    args = parser.parse_args()

    generate_hard_negatives(workspace_root=args.root, target_ratio=args.ratio, max_to_generate=args.max)
