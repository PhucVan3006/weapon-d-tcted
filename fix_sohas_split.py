"""
fix_sohas_split.py - Sửa lỗi process_sohas.py đổ dữ liệu SOHAS vào tập TEST.

VẤN ĐỀ
------
process_sohas.py lặp qua [('train','train'), ('test','test')] nên 857 ảnh
thuộc split TEST của SOHAS (trong đó 320 ảnh hard negative) bị copy thẳng
vào data/dataset_merged/test/. Hai hậu quả, cả hai đều nặng:

  1. TẬP TEST BỊ ĐỔI. Mốc cũ mAP50 = 0.8504 đo trên tập test gốc. Sau khi
     trộn thêm một dataset khác vào, con số mới KHÔNG còn so sánh được với
     mốc đó. Mất luôn thước đo để biết lần train sau tốt hơn hay tệ hơn.

  2. DỮ LIỆU TỐT NHẤT BỊ ĐEM ĐI ĐO THAY VÌ ĐEM ĐI HỌC. Split test của
     SOHAS là phần đúng miền nhất của cả bộ: tên file dạng ABbframe00145,
     MBsframe..., KravMagaTraining... cho thấy đây là frame trích từ video
     thật, và trung vị diện tích hộp chỉ 2-5% khung (so với 43% ở split
     train). Đó chính là loại ảnh hệ thống đang thiếu. Để nó ở test thì
     mô hình không bao giờ được học từ nó.

CÁCH SỬA
--------
Chuyển toàn bộ file sohas_* ra khỏi test, chia:
    85% -> train    (để học)
    15% -> valid    (để chọn ngưỡng và theo dõi overfit)
Tập test trở lại chỉ còn dữ liệu gốc của dự án -> mốc so sánh được giữ.

Chia theo băm MD5 của tên file nên KHÔNG đổi giữa các lần chạy, và ảnh
cùng tên luôn vào cùng một phía.

CÁCH DÙNG
---------
    python fix_sohas_split.py            # xem trước, không đổi gì
    python fix_sohas_split.py --apply    # thực hiện
    python fix_sohas_split.py --undo     # trả về đúng chỗ cũ theo manifest
"""

import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MERGED = PROJECT_ROOT / "data" / "dataset_merged"
MANIFEST = PROJECT_ROOT / "data" / "fix_sohas_split_manifest.json"

PREFIX = "sohas_"
VALID_PCT = 15          # phần trăm đưa sang valid
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rel(p: Path) -> str:
    """
    Đường dẫn TƯƠNG ĐỐI so với thư mục dự án, để ghi vào manifest.

    VÌ SAO: bản đầu ghi đường dẫn TUYỆT ĐỐI. Nếu thư mục dự án bị đổi tên
    hoặc di chuyển, --undo sẽ không tìm thấy file, hoặc tệ hơn là xoá/di
    chuyển file ở thư mục CŨ - lỗi này đã xảy ra thật khi tôi chạy --undo
    trong một bản copy và nó tác động lên bản gốc.
    """
    return str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def unrel(s: str) -> Path:
    """Dựng lại đường dẫn tuyệt đối và CHẶN mọi đường ra ngoài dự án."""
    q = (PROJECT_ROOT / s).resolve()
    if PROJECT_ROOT not in q.parents and q != PROJECT_ROOT:
        raise ValueError(f"Manifest trỏ ra ngoài thư mục dự án: {s}")
    return q


def bucket(stem: str) -> int:
    """0-99, cố định theo tên file nên lần chạy nào cũng cho cùng kết quả."""
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest()[:8], 16) % 100


def label_is_empty(p: Path) -> bool:
    try:
        return not any(l.strip() for l in p.read_text(encoding="utf-8").splitlines())
    except Exception:
        return False


def collect():
    """Tìm mọi cặp ảnh/nhãn sohas_* đang nằm trong test."""
    img_dir = MERGED / "test" / "images"
    lbl_dir = MERGED / "test" / "labels"
    if not img_dir.is_dir():
        print(f"[!] Không thấy {img_dir}")
        return []

    items = []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXT:
            continue
        if not img.name.startswith(PREFIX):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        items.append({
            "stem": img.stem,
            "img": img,
            "lbl": lbl if lbl.exists() else None,
            "empty": lbl.exists() and label_is_empty(lbl),
            "dest": "valid" if bucket(img.stem) < VALID_PCT else "train",
        })
    return items


def count_split(split: str):
    """(số ảnh, số nhãn rỗng) của một split."""
    img_dir = MERGED / split / "images"
    lbl_dir = MERGED / split / "labels"
    if not img_dir.is_dir():
        return 0, 0
    n_img = sum(1 for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT)
    n_bg = 0
    if lbl_dir.is_dir():
        n_bg = sum(1 for p in lbl_dir.iterdir()
                   if p.suffix.lower() == ".txt" and label_is_empty(p))
    return n_img, n_bg


def report_splits(title):
    print(f"\n  {title}")
    print(f"  {'split':<10}{'số ảnh':>10}{'ảnh nền':>10}{'% nền':>9}")
    for s in ("train", "valid", "test"):
        n, bg = count_split(s)
        pct = (bg / n * 100) if n else 0.0
        print(f"  {s:<10}{n:>10}{bg:>10}{pct:>8.1f}%")


def move_one(src: Path, dst_dir: Path, apply: bool):
    """Di chuyển, tự đổi tên nếu đích đã có file cùng tên."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        i = 2
        while (dst_dir / f"{src.stem}__{i}{src.suffix}").exists():
            i += 1
        dst = dst_dir / f"{src.stem}__{i}{src.suffix}"
    if apply:
        shutil.move(str(src), str(dst))
    return dst


def do_fix(apply: bool):
    items = collect()
    if not items:
        print("[i] Không thấy file sohas_* nào trong test. Có thể đã sửa rồi,")
        print("    hoặc process_sohas.py chưa chạy với --apply.")
        return

    n_train = sum(1 for it in items if it["dest"] == "train")
    n_valid = len(items) - n_train
    n_bg = sum(1 for it in items if it["empty"])
    n_nolbl = sum(1 for it in items if it["lbl"] is None)

    print("=" * 74)
    print("   SỬA LỖI: CHUYỂN DỮ LIỆU SOHAS RA KHỎI TẬP TEST")
    print("=" * 74)
    print(f"   Tìm thấy trong test : {len(items)} ảnh có tiền tố '{PREFIX}'")
    print(f"     trong đó ảnh nền  : {n_bg} (nhãn rỗng - hard negative)")
    if n_nolbl:
        print(f"     [!] thiếu nhãn    : {n_nolbl} ảnh -> sẽ chuyển kèm nhãn rỗng")
    print(f"   Sẽ chuyển sang train: {n_train}")
    print(f"   Sẽ chuyển sang valid: {n_valid}")

    report_splits("TRƯỚC khi sửa:")

    if not apply:
        print("\n   Mẫu 8 file đầu:")
        for it in items[:8]:
            tag = "nền" if it["empty"] else "có vũ khí"
            print(f"     {it['img'].name:<46} -> {it['dest']:<6} ({tag})")
        print("\n[!] Đây là bản xem trước, chưa đổi gì.")
        print("    Chạy lại với --apply để thực hiện.")
        return

    moved = []
    for it in items:
        img_dst = move_one(it["img"], MERGED / it["dest"] / "images", True)
        lbl_dir = MERGED / it["dest"] / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        if it["lbl"] is not None:
            lbl_dst = move_one(it["lbl"], lbl_dir, True)
        else:
            # Ảnh không có nhãn -> tạo nhãn rỗng, coi là ảnh nền.
            # Để trống hẳn thì ultralytics vẫn tính là nền, nhưng ghi file
            # rõ ràng thì audit đếm được và không nhầm là thiếu nhãn.
            lbl_dst = lbl_dir / f"{img_dst.stem}.txt"
            lbl_dst.write_text("", encoding="utf-8")
        moved.append({
            "dest": it["dest"],
            "img_to": rel(img_dst), "img_from": rel(it["img"]),
            "lbl_to": rel(lbl_dst),
            "lbl_from": rel(it["lbl"]) if it["lbl"] else None,
        })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(moved, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print(f"\n[XONG] Đã chuyển {len(moved)} cặp ảnh/nhãn.")
    print(f"       Manifest để hoàn tác: {MANIFEST}")
    report_splits("SAU khi sửa:")
    print("\n   Bước tiếp theo:")
    print("     python audit_dataset.py --data data/weapon_8cls.yaml")


def do_undo():
    if not MANIFEST.exists():
        print(f"[!] Không thấy manifest {MANIFEST} - không biết trả về đâu.")
        return
    moved = json.loads(MANIFEST.read_text(encoding="utf-8"))
    n = 0
    removed = 0
    for m in moved:
        for key_to, key_from in (("img_to", "img_from"), ("lbl_to", "lbl_from")):
            src = unrel(m[key_to])
            dst = m[key_from]
            if not src.exists():
                continue
            if dst is None:
                # File này do chính script tạo ra (nhãn rỗng cho ảnh thiếu
                # nhãn), không có chỗ cũ để trả về. Phải XOÁ, chứ bỏ lại thì
                # thành nhãn mồ côi không còn ảnh đi kèm.
                src.unlink()
                removed += 1
                continue
            dst = unrel(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            n += 1
    MANIFEST.unlink()
    print(f"[XONG] Đã trả {n} file về tập test"
          + (f", xoá {removed} nhãn rỗng do script tự tạo" if removed else "")
          + " và xoá manifest.")
    report_splits("Sau khi hoàn tác:")


def main():
    ap = argparse.ArgumentParser(
        description="Chuyển dữ liệu SOHAS ra khỏi tập test, chia về train/valid")
    ap.add_argument("--apply", action="store_true", help="thực hiện thật")
    ap.add_argument("--undo", action="store_true", help="hoàn tác theo manifest")
    args = ap.parse_args()

    if not MERGED.is_dir():
        print(f"[!] Không thấy {MERGED}")
        sys.exit(1)

    if args.undo:
        do_undo()
    else:
        do_fix(args.apply)


if __name__ == "__main__":
    main()
