"""
rebalance_dataset.py - Cân bằng lại dataset 12 lớp xuống 8 lớp vũ khí thật.

CĂN CỨ TỪ SỐ LIỆU THẬT (audit_dataset.py trên tập train 25.963 ảnh)
-------------------------------------------------------------------
    Chain    32.838 hộp  = 64,6% TOÀN BỘ dữ liệu
    Knife     4.121       =  8,1%
    ...
    Spanner     208       =  0,4%
    Stick        19       =  0,0%     <- mười chín hộp

Lệch lớp Chain/Stick = 1.728 lần. Gần hai phần ba tín hiệu huấn luyện đổ
vào lớp ÍT NGUY HIỂM NHẤT và gây BÁO NHẦM NHIỀU NHẤT (22% số báo nhầm
theo ma trận nhầm lẫn). Mô hình thành ra một máy dò dây xích.

BA LỚP KHÔNG THỂ HUẤN LUYỆN: Stick (19), Spanner (208), Hammer (428).
Không mô hình nào học nổi một lớp với ngần ấy mẫu; chúng chỉ làm nhiễu.

SCRIPT NÀY LÀM GÌ
-----------------
  1. Xoá nhãn của 4 lớp: Chain, Stick, Hammer, Spanner
  2. Ảnh nào mất hết nhãn -> trở thành ẢNH NỀN (nhãn rỗng). Đây là món
     quà: dataset đang thiếu ảnh nền trầm trọng (4,6%, cần 10-20%).
  3. Giữ lại đúng số ảnh nền cần thiết để đạt tỉ lệ mục tiêu, phần thừa
     chuyển sang thư mục riêng (KHÔNG xoá, lấy lại được).
  4. Sinh file data/weapon_8cls.yaml mới.

ID LỚP 0-7 GIỮ NGUYÊN, chỉ bỏ 8-11, nên không cần ánh xạ lại gì.

AN TOÀN
-------
  - Mặc định chỉ XEM TRƯỚC, không đụng file nào. Cần --apply mới chạy thật.
  - Sao lưu toàn bộ nhãn gốc vào labels_backup_12cls/ trước khi sửa.
  - Có --undo để hoàn tác.

CÁCH DÙNG
---------
    python rebalance_dataset.py                 # xem trước
    python rebalance_dataset.py --apply         # thực hiện
    python rebalance_dataset.py --undo          # hoàn tác
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter

try:
    import yaml
except ImportError:
    yaml = None

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

OLD_NAMES = ['Axe', 'Sickle', 'Sword', 'Crowbar', 'Knife', 'Pistol',
             'Long_gun', 'Grenade', 'Stick', 'Hammer', 'Spanner', 'Chain']

# Bỏ 4 lớp cuối. ID 0-7 của các lớp giữ lại không thay đổi.
DROP_IDS = {8, 9, 10, 11}           # Stick, Hammer, Spanner, Chain
KEEP_NAMES = OLD_NAMES[:8]

BACKUP_DIR = "labels_backup_12cls"
SURPLUS_DIR = "surplus_background"
TARGET_BG_RATIO = 0.15              # 15% ảnh nền


def split_dirs(root, split):
    base = Path(root) / split
    return base / "images", base / "labels"


def read_boxes(p):
    out = []
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        out.append((int(float(parts[0])), parts[1:5]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return out


def analyse(root, splits):
    """Xem trước: sau khi bỏ 4 lớp thì dataset ra sao."""
    report = {}
    for sp in splits:
        img_dir, lbl_dir = split_dirs(root, sp)
        if not img_dir.is_dir():
            continue
        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]

        kept_cnt, dropped_cnt = Counter(), Counter()
        now_bg = 0          # vốn đã là ảnh nền
        will_bg = 0         # sẽ thành ảnh nền sau khi bỏ lớp
        still_obj = 0

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            boxes = read_boxes(lbl) if lbl.exists() else []
            if not boxes:
                now_bg += 1
                continue
            keep = [b for b in boxes if b[0] not in DROP_IDS]
            for cid, _ in boxes:
                (dropped_cnt if cid in DROP_IDS else kept_cnt)[cid] += 1
            if keep:
                still_obj += 1
            else:
                will_bg += 1

        report[sp] = {
            "n_img": len(images), "kept": kept_cnt, "dropped": dropped_cnt,
            "now_bg": now_bg, "will_bg": will_bg, "still_obj": still_obj,
        }
    return report


def print_preview(report):
    for sp, r in report.items():
        n_keep = sum(r["kept"].values())
        n_drop = sum(r["dropped"].values())
        total_bg = r["now_bg"] + r["will_bg"]

        print("\n" + "=" * 78)
        print(f"   [{sp.upper()}]  {r['n_img']} ảnh")
        print("=" * 78)
        print(f"   Hộp giữ lại : {n_keep}")
        print(f"   Hộp xoá đi  : {n_drop}  "
              f"({', '.join(f'{OLD_NAMES[c]}={n}' for c, n in sorted(r['dropped'].items()))})")
        print()
        print(f"   Ảnh còn vật thể      : {r['still_obj']}")
        print(f"   Ảnh nền sẵn có       : {r['now_bg']}")
        print(f"   Ảnh THÀNH ảnh nền    : {r['will_bg']}   <- quà từ việc bỏ Chain")
        print(f"   Tổng ảnh nền         : {total_bg} "
              f"({total_bg / r['n_img'] * 100:.1f}% nếu giữ hết)")

        if n_keep:
            print(f"\n   {'Lớp':<12}{'số hộp':>9}{'% mới':>9}")
            print("   " + "-" * 30)
            for cid in sorted(r["kept"], key=lambda c: -r["kept"][c]):
                print(f"   {OLD_NAMES[cid]:<12}{r['kept'][cid]:>9}"
                      f"{r['kept'][cid] / n_keep * 100:>8.1f}%")
            print("   " + "-" * 30)
            mx, mn = max(r["kept"].values()), min(r["kept"].values())
            print(f"   Lệch lớp sau khi cân bằng: {mx / mn:.1f} lần")

        # Gợi ý số ảnh nền nên giữ
        if sp == "train" and r["still_obj"]:
            want = int(r["still_obj"] * TARGET_BG_RATIO / (1 - TARGET_BG_RATIO))
            print(f"\n   Để đạt {TARGET_BG_RATIO*100:.0f}% ảnh nền, nên giữ "
                  f"~{want} ảnh nền; phần thừa chuyển sang {SURPLUS_DIR}/")


def apply_changes(root, splits, target_bg_ratio, seed=42):
    import random
    rnd = random.Random(seed)
    root = Path(root)

    for sp in splits:
        img_dir, lbl_dir = split_dirs(root, sp)
        if not img_dir.is_dir():
            continue

        # --- Sao lưu nhãn gốc ---
        backup = root / sp / BACKUP_DIR
        if not backup.exists():
            print(f"[+] Sao lưu nhãn gốc {sp} -> {backup}")
            shutil.copytree(lbl_dir, backup)
        else:
            print(f"[!] {backup} đã tồn tại -> đã chạy trước đó. Bỏ qua {sp}.")
            print(f"    Dùng --undo nếu muốn làm lại.")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        became_bg, kept_obj = [], []

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            boxes = read_boxes(lbl) if lbl.exists() else []
            keep = [b for b in boxes if b[0] not in DROP_IDS]

            with open(lbl, "w", encoding="utf-8") as f:
                for cid, coords in keep:
                    f.write(f"{cid} {' '.join(coords)}\n")

            (kept_obj if keep else became_bg).append(img)

        # --- Khống chế tỉ lệ ảnh nền ---
        if sp == "train" and kept_obj:
            want = int(len(kept_obj) * target_bg_ratio / (1 - target_bg_ratio))
            if len(became_bg) > want:
                rnd.shuffle(became_bg)
                surplus = became_bg[want:]
                sdir_i = root / sp / SURPLUS_DIR / "images"
                sdir_l = root / sp / SURPLUS_DIR / "labels"
                sdir_i.mkdir(parents=True, exist_ok=True)
                sdir_l.mkdir(parents=True, exist_ok=True)
                for img in surplus:
                    lbl = lbl_dir / f"{img.stem}.txt"
                    shutil.move(str(img), str(sdir_i / img.name))
                    if lbl.exists():
                        shutil.move(str(lbl), str(sdir_l / lbl.name))
                print(f"[+] {sp}: chuyển {len(surplus)} ảnh nền thừa sang {SURPLUS_DIR}/")
                print(f"    Giữ lại {want} ảnh nền "
                      f"({want/(want+len(kept_obj))*100:.1f}%)")

        print(f"[+] {sp}: xong — {len(kept_obj)} ảnh có vật thể")

    # --- Xoá cache cũ ---
    for c in root.rglob("*.cache"):
        try:
            c.unlink()
            print(f"[+] Đã xoá cache cũ: {c.name}")
        except Exception:
            pass


def write_yaml(root, out_path):
    txt = (
        "# Weapon Detection Dataset - 8 lớp vũ khí thật\n"
        "# Sinh bởi rebalance_dataset.py\n"
        "#\n"
        "# Đã bỏ 4 lớp khỏi bản 12 lớp:\n"
        "#   Chain   (32.838 hộp = 64,6% dữ liệu, gây 22% số báo nhầm)\n"
        "#   Stick   (19 hộp - không đủ để huấn luyện)\n"
        "#   Hammer  (428 hộp)\n"
        "#   Spanner (208 hộp)\n"
        "# ID lớp 0-7 giữ nguyên nên trọng số cũ vẫn tương thích phần lớn.\n\n"
        f"path: {root}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 8\n"
        f"names: {KEEP_NAMES}\n"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(txt, encoding="utf-8")
    print(f"\n[+] Đã ghi cấu hình mới: {out_path}")


def undo(root, splits):
    root = Path(root)
    n = 0
    for sp in splits:
        base = root / sp
        backup, lbl_dir = base / BACKUP_DIR, base / "labels"
        if not backup.is_dir():
            print(f"[!] {sp}: không có bản sao lưu, bỏ qua.")
            continue

        # Trả ảnh nền thừa về chỗ cũ
        sdir = base / SURPLUS_DIR
        if sdir.is_dir():
            for sub, dst in (("images", base / "images"), ("labels", lbl_dir)):
                src = sdir / sub
                if src.is_dir():
                    for f in src.iterdir():
                        shutil.move(str(f), str(dst / f.name))
            shutil.rmtree(sdir, ignore_errors=True)
            print(f"[+] {sp}: đã trả ảnh nền thừa về chỗ cũ")

        shutil.rmtree(lbl_dir, ignore_errors=True)
        shutil.move(str(backup), str(lbl_dir))
        print(f"[+] {sp}: đã khôi phục nhãn 12 lớp")
        n += 1

    for c in root.rglob("*.cache"):
        try:
            c.unlink()
        except Exception:
            pass
    print(f"\n[XONG] Đã hoàn tác {n} tập.")


def main():
    ap = argparse.ArgumentParser(
        description="Cân bằng dataset: 12 lớp -> 8 lớp vũ khí thật")
    ap.add_argument("--root", default="data/dataset_merged")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--bg-ratio", type=float, default=TARGET_BG_RATIO)
    ap.add_argument("--out-yaml", default="data/weapon_8cls.yaml")
    ap.add_argument("--apply", action="store_true", help="Thực hiện thật")
    ap.add_argument("--undo", action="store_true", help="Hoàn tác")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    root = args.root

    if not Path(root).is_dir():
        print(f"[!] Không tìm thấy thư mục dataset: {root}")
        return

    print("=" * 78)
    print("   CÂN BẰNG LẠI DATASET: 12 LỚP -> 8 LỚP VŨ KHÍ THẬT")
    print(f"   Thư mục: {root}")
    print("=" * 78)
    print("   Bỏ: " + ", ".join(OLD_NAMES[i] for i in sorted(DROP_IDS)))
    print("   Giữ: " + ", ".join(KEEP_NAMES))

    if args.undo:
        undo(root, splits)
        return

    print_preview(analyse(root, splits))

    if not args.apply:
        print("\n" + "=" * 78)
        print("   ĐÂY MỚI LÀ XEM TRƯỚC — chưa có file nào bị thay đổi.")
        print("   Chạy lại kèm --apply để thực hiện:")
        print("     python rebalance_dataset.py --apply")
        print("=" * 78)
        return

    print("\n" + "=" * 78)
    apply_changes(root, splits, args.bg_ratio)
    write_yaml(os.path.abspath(root).replace("\\", "/"), args.out_yaml)

    print("\n" + "=" * 78)
    print("   BƯỚC TIẾP THEO")
    print("=" * 78)
    print("   1. Kiểm tra lại phân bố mới:")
    print(f"      python audit_dataset.py --data {args.out_yaml}")
    print("   2. Huấn luyện lại trên 8 lớp:")
    print(f"      python src/training/train.py --mode full --data {args.out_yaml}")
    print("   3. Nếu muốn quay lại 12 lớp:")
    print("      python rebalance_dataset.py --undo")
    print("=" * 78)


if __name__ == "__main__":
    main()
