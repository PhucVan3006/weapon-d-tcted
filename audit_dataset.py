"""
audit_dataset.py - Kiểm định toàn diện bộ dữ liệu trước khi huấn luyện.

VÌ SAO QUAN TRỌNG
-----------------
Hệ thống đang vẽ hộp lên CÁNH TAY và BALO. Nguyên nhân gần như chắc chắn
nằm trong chính bộ dữ liệu chứ không phải thuật toán:

  - Nếu đa số hộp trong tập train rất TO (vật thể làm chủ thể của ảnh),
    mô hình học rằng "vật to ở giữa khung = vũ khí" và sẽ bám vào thân
    người khi gặp ảnh đời thật.
  - Nếu KHÔNG có ảnh nền (nhãn rỗng), mô hình chưa từng được dạy thế nào
    là "khung hình không có gì", nên nó luôn cố tìm ra một thứ gì đó.
  - Nếu phân bố lớp lệch nặng, lớp ít mẫu sẽ vừa bỏ sót vừa báo nhầm.

Script này đo tất cả những điều đó và nói rõ cần sửa gì.

CÁCH DÙNG
---------
    python audit_dataset.py
    python audit_dataset.py --data data/weapon_12cls.yaml
    python audit_dataset.py --check-leak     # thêm kiểm tra trùng ảnh (chậm hơn)
"""

import os
import sys
import glob
import math
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

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

# Ngưỡng cảnh báo
BG_RATIO_MIN = 0.08          # dưới 8% ảnh nền là quá ít
BG_RATIO_MAX = 0.30          # trên 30% là quá nhiều, kéo recall xuống
HUGE_BOX_PCT = 25.0          # hộp chiếm >25% khung bị coi là "vật thể làm chủ thể"
TINY_BOX_PCT = 0.05          # hộp <0.05% khung gần như không học được
IMBALANCE_WARN = 8.0         # lệch lớp quá 8 lần là đáng lo


def pct(n, d):
    return (n / d * 100.0) if d else 0.0


def load_data_yaml(path):
    if yaml is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_dir(root, rel):
    p = Path(rel)
    if not p.is_absolute():
        p = Path(root) / rel
    return p


def read_label(path):
    """Trả về danh sách (cls_id, xc, yc, w, h). Hộp hỏng bị bỏ qua."""
    out, bad = [], 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    bad += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    bad += 1
                    continue
                if not (0 < w <= 1.5 and 0 < h <= 1.5):
                    bad += 1
                    continue
                out.append((cid, xc, yc, w, h))
    except Exception:
        bad += 1
    return out, bad


def audit_split(split_name, img_dir, names):
    img_dir = Path(img_dir)
    lbl_dir = Path(str(img_dir).replace("images", "labels"))

    if not img_dir.is_dir():
        print(f"\n[!] Không tìm thấy thư mục ảnh: {img_dir}")
        return None

    images = [p for p in img_dir.iterdir()
              if p.suffix.lower() in IMG_EXT]
    n_img = len(images)
    if n_img == 0:
        print(f"\n[!] {split_name}: không có ảnh nào.")
        return None

    cls_count = Counter()
    cls_images = defaultdict(set)
    areas = defaultdict(list)       # cls -> % diện tích khung
    all_areas = []
    n_bg = 0
    n_missing_lbl = 0
    n_bad_lines = 0
    huge, tiny = 0, 0
    huge_by_cls = Counter()

    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            n_missing_lbl += 1
            continue
        boxes, bad = read_label(lbl)
        n_bad_lines += bad
        if not boxes:
            n_bg += 1
            continue
        for cid, xc, yc, w, h in boxes:
            cls_count[cid] += 1
            cls_images[cid].add(img.stem)
            a = w * h * 100.0
            areas[cid].append(a)
            all_areas.append(a)
            if a > HUGE_BOX_PCT:
                huge += 1
                huge_by_cls[cid] += 1
            elif a < TINY_BOX_PCT:
                tiny += 1

    n_box = sum(cls_count.values())
    print("\n" + "=" * 86)
    print(f"   [{split_name.upper()}]  {n_img} ảnh  |  {n_box} hộp bao")
    print("=" * 86)
    print(f"   Ảnh nền (nhãn rỗng) : {n_bg} ({pct(n_bg, n_img):.1f}%)")
    if n_missing_lbl:
        print(f"   Ảnh THIẾU file nhãn : {n_missing_lbl}  <-- ultralytics coi như ảnh nền")
    if n_bad_lines:
        print(f"   Dòng nhãn hỏng      : {n_bad_lines}")

    if n_box:
        print(f"\n   {'Lớp':<12}{'số hộp':>9}{'số ảnh':>9}{'% hộp':>8}"
              f"{'DT trung vị':>13}{'DT p90':>9}{'hộp>25%':>10}")
        print("   " + "-" * 80)
        for cid in sorted(cls_count, key=lambda c: -cls_count[c]):
            a = sorted(areas[cid])
            med = a[len(a) // 2] if a else 0
            p90 = a[int(len(a) * 0.9)] if a else 0
            nm = names.get(cid, f"class_{cid}")
            print(f"   {nm:<12}{cls_count[cid]:>9}{len(cls_images[cid]):>9}"
                  f"{pct(cls_count[cid], n_box):>8.1f}{med:>13.2f}{p90:>9.2f}"
                  f"{huge_by_cls[cid]:>10}")
        print("   " + "-" * 80)

        srt = sorted(all_areas)
        print(f"   Diện tích hộp toàn tập: trung vị {srt[len(srt)//2]:.2f}% khung  |  "
              f"p90 {srt[int(len(srt)*0.9)]:.2f}%  |  p99 {srt[int(len(srt)*0.99)]:.2f}%")
        print(f"   Hộp quá TO (>{HUGE_BOX_PCT}% khung): {huge} ({pct(huge, n_box):.1f}%)")
        print(f"   Hộp quá NHỎ (<{TINY_BOX_PCT}% khung): {tiny} ({pct(tiny, n_box):.1f}%)")

    return {
        "split": split_name, "n_img": n_img, "n_box": n_box, "n_bg": n_bg,
        "cls_count": cls_count, "areas": areas, "all_areas": all_areas,
        "huge": huge, "tiny": tiny, "missing": n_missing_lbl,
        "images": images,
    }


def check_leak(stats):
    """Kiểm tra ảnh trùng giữa các split bằng mã băm nội dung."""
    print("\n" + "=" * 86)
    print("   KIỂM TRA RÒ RỈ DỮ LIỆU GIỮA CÁC TẬP")
    print("=" * 86)
    hashes = {}
    for s in stats:
        if not s:
            continue
        for img in s["images"]:
            try:
                h = hashlib.md5(img.read_bytes()).hexdigest()
            except Exception:
                continue
            hashes.setdefault(h, []).append((s["split"], img.name))

    leaks = {h: v for h, v in hashes.items()
             if len({sp for sp, _ in v}) > 1}
    dups = {h: v for h, v in hashes.items()
            if len(v) > 1 and len({sp for sp, _ in v}) == 1}

    if leaks:
        print(f"   [NGHIÊM TRỌNG] {len(leaks)} ảnh xuất hiện ở NHIỀU tập khác nhau.")
        print("   Điều này làm điểm đánh giá CAO GIẢ TẠO - mô hình đã thấy ảnh test")
        print("   trong lúc train. Phải chia lại dataset.")
        for h, v in list(leaks.items())[:5]:
            print("     " + "  <->  ".join(f"{sp}/{nm}" for sp, nm in v))
    else:
        print("   Không có ảnh nào trùng giữa train/valid/test. Tốt.")

    if dups:
        print(f"\n   {len(dups)} nhóm ảnh trùng lặp TRONG CÙNG một tập "
              f"(làm mô hình học vẹt, nên loại bớt).")


def verdict(stats, names):
    print("\n" + "=" * 86)
    print("   KẾT LUẬN & VIỆC CẦN LÀM")
    print("=" * 86)

    train = next((s for s in stats if s and s["split"] == "train"), None)
    if not train:
        print("   Không đọc được tập train.")
        return

    issues = []

    # 1. Ảnh nền
    bg = pct(train["n_bg"], train["n_img"])
    if bg < BG_RATIO_MIN * 100:
        issues.append((
            "NGHIÊM TRỌNG",
            f"Tập train chỉ có {bg:.1f}% ảnh nền (nhãn rỗng).",
            "Đây là nguyên nhân TRỰC TIẾP của việc vẽ hộp lên cánh tay và balo. "
            "Mô hình chưa từng được dạy thế nào là khung hình KHÔNG có vũ khí, "
            "nên nó luôn cố tìm ra thứ gì đó.\n"
            "       -> Chạy: python collect_hard_negatives.py\n"
            "          Mục tiêu: nâng tỉ lệ ảnh nền lên 10-20%."
        ))
    elif bg > BG_RATIO_MAX * 100:
        issues.append((
            "CẢNH BÁO",
            f"Tập train có tới {bg:.1f}% ảnh nền.",
            "Quá nhiều ảnh nền dạy mô hình 'không thấy gì' -> recall tụt.\n"
            "       -> Giảm bớt về khoảng 15-20%."
        ))

    # 2. Kích thước hộp
    if train["all_areas"]:
        srt = sorted(train["all_areas"])
        med = srt[len(srt) // 2]
        huge_pct = pct(train["huge"], train["n_box"])
        if med > 15.0 or huge_pct > 25.0:
            issues.append((
                "NGHIÊM TRỌNG",
                f"Hộp trong tập train quá TO: trung vị {med:.1f}% khung, "
                f"{huge_pct:.0f}% số hộp chiếm hơn {HUGE_BOX_PCT}% khung.",
                "Dataset gồm ảnh CHỤP VẬT THỂ LÀM CHỦ THỂ, không phải ảnh giám sát. "
                "Mô hình học rằng 'vật to giữa khung = vũ khí', nên khi gặp video "
                "thật nó bám vào thân người, cánh tay, balo.\n"
                "       -> Hai cách chữa, nên làm CẢ HAI:\n"
                "          a) Thêm hard negative từ video thật (rẻ, không cần gán nhãn)\n"
                "          b) Gán nhãn 200-500 khung CÓ vũ khí từ video thật, trộn vào"
            ))

    # 3. Mất cân bằng lớp
    cc = train["cls_count"]
    if len(cc) >= 2:
        mx, mn = max(cc.values()), min(cc.values())
        ratio = mx / mn if mn else float("inf")
        if ratio > IMBALANCE_WARN:
            top = names.get(max(cc, key=cc.get), "?")
            bot = names.get(min(cc, key=cc.get), "?")
            issues.append((
                "CẢNH BÁO",
                f"Lệch lớp {ratio:.1f} lần ({top}={mx} so với {bot}={mn}).",
                "Lớp ít mẫu vừa bỏ sót nhiều vừa hay bị nhầm sang lớp khác.\n"
                "       -> Cân nhắc gộp lớp hiếm, hoặc bổ sung ảnh cho lớp đó."
            ))

    # 4. Thiếu nhãn
    if train["missing"]:
        issues.append((
            "CẢNH BÁO",
            f"{train['missing']} ảnh train không có file nhãn.",
            "Ultralytics coi chúng là ảnh nền. Nếu thực tế chúng CÓ vũ khí thì "
            "bạn đang dạy mô hình bỏ qua vũ khí.\n"
            "       -> Kiểm tra lại, hoặc xoá hẳn những ảnh đó."
        ))

    if not issues:
        print("\n   Không phát hiện vấn đề nghiêm trọng nào trong dataset.")
    else:
        for i, (level, title, detail) in enumerate(issues, 1):
            print(f"\n   {i}. [{level}] {title}")
            print(f"       {detail}")

    print("\n" + "=" * 86)


def main():
    ap = argparse.ArgumentParser(description="Kiểm định bộ dữ liệu YOLO")
    ap.add_argument("--data", default="data/weapon_12cls.yaml")
    ap.add_argument("--check-leak", action="store_true",
                    help="Kiểm tra ảnh trùng giữa các tập (chậm, đọc toàn bộ ảnh)")
    args = ap.parse_args()

    cfg = load_data_yaml(args.data)
    if not cfg:
        print(f"[!] Không đọc được {args.data}")
        return

    root = cfg.get("path", ".")
    raw_names = cfg.get("names", [])
    names = raw_names if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    names = {int(k): v for k, v in names.items()}

    print("=" * 86)
    print("   KIỂM ĐỊNH BỘ DỮ LIỆU")
    print(f"   Cấu hình : {args.data}")
    print(f"   Thư mục  : {root}")
    print(f"   Số lớp   : {cfg.get('nc')} — {', '.join(names.values())}")
    print("=" * 86)

    stats = []
    for key, label in (("train", "train"), ("val", "valid"), ("test", "test")):
        rel = cfg.get(key)
        if not rel:
            continue
        stats.append(audit_split(label, resolve_split_dir(root, rel), names))

    # Phân bố lớp giữa các tập có nhất quán không
    valid_stats = [s for s in stats if s and s["n_box"]]
    if len(valid_stats) >= 2:
        print("\n" + "=" * 86)
        print("   PHÂN BỐ LỚP GIỮA CÁC TẬP (phải TƯƠNG ĐỒNG nhau)")
        print("=" * 86)
        header = f"   {'Lớp':<12}" + "".join(f"{s['split']:>12}" for s in valid_stats)
        print(header)
        print("   " + "-" * (12 + 12 * len(valid_stats)))
        all_cls = sorted({c for s in valid_stats for c in s["cls_count"]})
        for cid in all_cls:
            row = f"   {names.get(cid, cid):<12}"
            for s in valid_stats:
                row += f"{pct(s['cls_count'].get(cid, 0), s['n_box']):>11.1f}%"
            print(row)
        print("   " + "-" * (12 + 12 * len(valid_stats)))
        print("   Cột lệch nhau nhiều nghĩa là chia tập không phân tầng -> điểm")
        print("   đánh giá không phản ánh đúng năng lực mô hình.")

    if args.check_leak:
        check_leak(stats)

    verdict(stats, names)


if __name__ == "__main__":
    main()
