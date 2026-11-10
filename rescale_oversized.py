"""
rescale_oversized.py - Thu nhỏ ảnh có hộp quá to rồi dán vào ảnh nền thật.

VẤN ĐỀ NÓ CHỮA
--------------
Audit tập train cho thấy:
    Axe     trung vị hộp 55.97% khung, p90 97.19%
    Sickle  trung vị hộp 47.59% khung
    toàn tập: 32.4% số hộp chiếm hơn 25% khung hình

Nghĩa là một nửa số hộp Axe chiếm hơn NỬA khung hình. Dataset này là ảnh
CHỤP VẬT THỂ LÀM CHỦ THỂ, không phải ảnh giám sát. Mô hình học đúng cái
điều sai: "vật to dài giữa khung = vũ khí". Khi gặp video CCTV thật, thứ to
và dài giữa khung lại là cánh tay, thân người và balo.

`scale=0.5` và `mosaic=1.0` lúc train đã cố chữa nhưng không đủ: mosaic chỉ
ghép 4 ảnh nên vật còn ~14% khung, trong khi trên camera giám sát thật vũ
khí chỉ chiếm 1-8%.

CÁCH LÀM
--------
Với mỗi ảnh train có hộp lớn nhất vượt 25% khung:
  1. Thu nhỏ cả ảnh theo hệ số f sao cho hộp lớn nhất còn 3-12% khung.
  2. Chọn MỘT ảnh nền thật từ chính kho hard negative đang có (4287 ảnh
     nhãn rỗng trong tập train).
  3. Dán ảnh đã thu nhỏ vào vị trí ngẫu nhiên trên ảnh nền.
  4. Tính lại nhãn theo tỉ lệ và độ lệch.
Ảnh GỐC được giữ nguyên, không xoá. Mô hình thấy cả hai cỡ.

CHỐNG RÒ RỈ TÍN HIỆU — phần quan trọng, đừng bỏ
-----------------------------------------------
Dán một ảnh chữ nhật lên ảnh khác để lại ĐƯỜNG GHÉP. Nếu đường ghép chỉ
xuất hiện ở ảnh CÓ vũ khí, mô hình sẽ học "thấy đường ghép là có vũ khí" -
tức là ta vừa dạy nó một đặc trưng giả, còn tệ hơn vấn đề ban đầu.

Hai biện pháp, dùng cả hai:
  a) Làm mờ biên dán (feather) nên không còn mép cứng.
  b) Sinh thêm một tỉ lệ ảnh ghép ÂM: lấy một ảnh nền thu nhỏ dán vào một
     ảnh nền khác, nhãn RỖNG. Đường ghép khi đó xuất hiện ở cả ảnh có và
     không có vũ khí, nên nó không còn mang thông tin gì.

CÁCH DÙNG
---------
    python rescale_oversized.py              # xem trước, không ghi gì
    python rescale_oversized.py --apply
    python rescale_oversized.py --undo
    python rescale_oversized.py --apply --limit 2000
"""

import sys
import json
import random
import hashlib
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MERGED = PROJECT_ROOT / "data" / "dataset_merged"
SPLIT = "train"          # CHỈ tập train. Augment tập valid/test là tự lừa mình.
MANIFEST = PROJECT_ROOT / "data" / "rescale_oversized_manifest.json"

PREFIX = "rescaled_"
OVERSIZE_PCT = 25.0      # ảnh có hộp lớn nhất vượt mức này mới được xử lý
TARGET_MIN_PCT = 3.0     # cỡ mục tiêu của hộp lớn nhất sau khi thu
TARGET_MAX_PCT = 12.0
MIN_KEEP_PCT = 0.03      # hộp nhỏ hơn mức này sau khi thu thì bỏ
NEG_RATIO = 0.25         # số ảnh ghép ÂM / số ảnh ghép dương
FEATHER = 10             # độ rộng dải làm mờ biên, tính theo pixel
JPEG_Q = 92

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------
def rel(p) -> str:
    """Đường dẫn tương đối so với thư mục dự án - xem ghi chú trong
    fix_sohas_split.py về lý do không dùng đường dẫn tuyệt đối."""
    return str(Path(p).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def unrel(s: str) -> Path:
    """Dựng lại đường dẫn và CHẶN mọi đường ra ngoài dự án."""
    q = (PROJECT_ROOT / s).resolve()
    if PROJECT_ROOT not in q.parents and q != PROJECT_ROOT:
        raise ValueError(f"Manifest trỏ ra ngoài thư mục dự án: {s}")
    return q


def load_manifest():
    if not MANIFEST.exists():
        return []
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_manifest(entries):
    """
    GỘP với bản ghi cũ, KHÔNG ghi đè.

    VÌ SAO: bản đầu ghi đè. Chạy --apply lần thứ hai (khi không còn gì để
    sinh) sẽ ghi một danh sách RỖNG lên manifest, xoá sạch bản ghi của lần
    một -> --undo sau đó không xoá được gì và ảnh sinh ra nằm lại vĩnh viễn.
    """
    old = load_manifest()
    seen = {tuple(e) for e in old}
    merged = old + [e for e in entries if tuple(e) not in seen]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    return len(merged)


def rng_for(name: str) -> random.Random:
    """RNG cố định theo tên file -> chạy lại cho kết quả y như cũ."""
    seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:12], 16)
    return random.Random(seed)


def read_labels(p: Path):
    """[(cls, xc, yc, w, h)] theo chuẩn YOLO đã chuẩn hoá. Rỗng = ảnh nền."""
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(float(parts[0])), *[float(v) for v in parts[1:5]]))
        except ValueError:
            continue
    return out


def write_labels(p: Path, rows):
    p.write_text(
        "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows),
        encoding="utf-8")


def fit_background(bg, W, H):
    """Phủ ảnh nền kín khung WxH: scale giữ tỉ lệ rồi cắt giữa."""
    bh, bw = bg.shape[:2]
    s = max(W / bw, H / bh)
    nb = cv2.resize(bg, (max(W, int(bw * s + 0.5)), max(H, int(bh * s + 0.5))),
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0 = (nb.shape[0] - H) // 2
    x0 = (nb.shape[1] - W) // 2
    return nb[y0:y0 + H, x0:x0 + W].copy()


def feather_paste(canvas, patch, ox, oy, width=FEATHER):
    """
    Dán patch lên canvas với biên mờ dần, để không còn mép chữ nhật cứng.
    Mép cứng là một đặc trưng giả mà mô hình có thể bám vào.
    """
    ph, pw = patch.shape[:2]
    if width <= 0 or ph <= 2 * width or pw <= 2 * width:
        canvas[oy:oy + ph, ox:ox + pw] = patch
        return
    ramp_y = np.ones((ph, 1), np.float32)
    ramp_x = np.ones((1, pw), np.float32)
    lin = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp_y[:width, 0] = lin
    ramp_y[-width:, 0] = lin[::-1]
    ramp_x[0, :width] = lin
    ramp_x[0, -width:] = lin[::-1]
    alpha = (ramp_y @ ramp_x)[:, :, None]
    region = canvas[oy:oy + ph, ox:ox + pw].astype(np.float32)
    canvas[oy:oy + ph, ox:ox + pw] = (
        patch.astype(np.float32) * alpha + region * (1.0 - alpha)
    ).astype(np.uint8)


def match_brightness(patch, canvas_region):
    """
    Kéo độ sáng trung bình của patch về gần ảnh nền. Không làm bước này thì
    một ô sáng/tối rõ rệt cũng thành đặc trưng giả như mép cứng.
    """
    pm = float(patch.reshape(-1, patch.shape[-1]).mean())
    cm = float(canvas_region.reshape(-1, canvas_region.shape[-1]).mean())
    if pm <= 1.0:
        return patch
    gain = np.clip(cm / pm, 0.75, 1.33)
    return np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
def scan():
    """Phân loại ảnh train: cần thu nhỏ / làm nền / bỏ qua."""
    img_dir = MERGED / SPLIT / "images"
    lbl_dir = MERGED / SPLIT / "labels"
    if not img_dir.is_dir():
        print(f"[!] Không thấy {img_dir}")
        return [], []

    oversized, backgrounds = [], []
    for img in sorted(img_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXT:
            continue
        if img.name.startswith(PREFIX):      # bản đã sinh lần trước
            continue
        rows = read_labels(lbl_dir / f"{img.stem}.txt")
        if not rows:
            backgrounds.append(img)
            continue
        max_area = max(w * h for _, _, _, w, h in rows) * 100.0
        if max_area > OVERSIZE_PCT:
            oversized.append((img, rows, max_area))
    return oversized, backgrounds


def make_one(src_img, rows, bg_path, out_img, out_lbl, r):
    """
    Sinh một ảnh ghép. Trả về số hộp giữ được, hoặc None nếu thất bại.
    rows = [] -> ảnh ghép ÂM, nhãn rỗng.
    """
    img = cv2.imread(str(src_img))
    bg = cv2.imread(str(bg_path))
    if img is None or bg is None:
        return None
    H, W = img.shape[:2]
    if W < 32 or H < 32:
        return None

    if rows:
        max_area = max(w * h for _, _, _, w, h in rows)
        target = r.uniform(TARGET_MIN_PCT, TARGET_MAX_PCT) / 100.0
        f = (target / max_area) ** 0.5
    else:
        # Ảnh âm: thu nhỏ theo hệ số ngẫu nhiên trong cùng dải để đường ghép
        # có cùng phân bố kích thước với ảnh dương.
        f = r.uniform(0.25, 0.60)
    f = min(f, 0.95)
    nw, nh = max(16, int(W * f)), max(16, int(H * f))
    if nw >= W or nh >= H:
        return None

    patch = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = fit_background(bg, W, H)
    ox = r.randint(0, W - nw)
    oy = r.randint(0, H - nh)
    patch = match_brightness(patch, canvas[oy:oy + nh, ox:ox + nw])
    feather_paste(canvas, patch, ox, oy)

    new_rows = []
    for c, xc, yc, bw, bh in rows:
        nxc = (xc * nw + ox) / W
        nyc = (yc * nh + oy) / H
        nbw = bw * nw / W
        nbh = bh * nh / H
        if nbw * nbh * 100.0 < MIN_KEEP_PCT:
            continue
        if not (0.0 < nxc < 1.0 and 0.0 < nyc < 1.0):
            continue
        new_rows.append((c, nxc, nyc, min(nbw, 1.0), min(nbh, 1.0)))

    if rows and not new_rows:
        return None       # thu nhỏ xong mất hết hộp -> bỏ, đừng tạo ảnh rác

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_img), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]):
        return None
    write_labels(out_lbl, new_rows)
    return len(new_rows)


# ----------------------------------------------------------------------
def report(oversized, backgrounds, n_neg):
    from collections import Counter
    print("=" * 76)
    print("   THU NHỎ ẢNH CÓ HỘP QUÁ TO RỒI DÁN VÀO ẢNH NỀN THẬT")
    print("=" * 76)
    print(f"   Tập xử lý            : {SPLIT} (chỉ train, không chạm valid/test)")
    print(f"   Ảnh có hộp > {OVERSIZE_PCT:.0f}%   : {len(oversized)}")
    print(f"   Kho ảnh nền sẵn có   : {len(backgrounds)} (nhãn rỗng)")
    print(f"   Sẽ sinh ảnh ghép DƯƠNG: {len(oversized)}")
    print(f"   Sẽ sinh ảnh ghép ÂM   : {n_neg}  (chống rò rỉ đường ghép)")
    print(f"   Cỡ hộp lớn nhất sau khi thu: {TARGET_MIN_PCT:.0f}-{TARGET_MAX_PCT:.0f}% khung")

    cnt = Counter()
    for _img, rows, _a in oversized:
        for c, _x, _y, _w, _h in rows:
            cnt[c] += 1
    if cnt:
        print("\n   Số hộp được thu nhỏ, theo class id:")
        for cid, n in sorted(cnt.items()):
            print(f"      id {cid}: {n}")
    if not backgrounds:
        print("\n   [!] KHÔNG có ảnh nền nào trong tập train -> không chạy được.")
        print("       Chạy fix_sohas_split.py --apply trước để có hard negative.")


def do_apply(limit):
    oversized, backgrounds = scan()
    if not oversized:
        print("[i] Không có ảnh nào vượt ngưỡng. Có thể đã chạy rồi.")
        return
    if limit:
        oversized = oversized[:limit]
    n_neg = int(len(oversized) * NEG_RATIO)
    report(oversized, backgrounds, n_neg)
    if not backgrounds:
        return

    img_dir = MERGED / SPLIT / "images"
    lbl_dir = MERGED / SPLIT / "labels"
    created, boxes, failed = [], 0, 0

    for i, (src, rows, _a) in enumerate(oversized, 1):
        r = rng_for(src.name)
        bg = backgrounds[r.randrange(len(backgrounds))]
        out_img = img_dir / f"{PREFIX}{src.stem}.jpg"
        out_lbl = lbl_dir / f"{PREFIX}{src.stem}.txt"
        if out_img.exists():
            continue
        n = make_one(src, rows, bg, out_img, out_lbl, r)
        if n is None:
            failed += 1
            continue
        boxes += n
        created.append([rel(out_img), rel(out_lbl)])
        if i % 500 == 0:
            print(f"      ... {i}/{len(oversized)}")

    # Ảnh ghép ÂM: nền thu nhỏ dán vào nền KHÁC, nhãn rỗng.
    n_neg_made = 0
    if len(backgrounds) < 2:
        print(f"[!] Chỉ có {len(backgrounds)} ảnh nền, không đủ để ghép ảnh âm.")
    else:
        for j in range(n_neg):
            r = rng_for(f"neg{j}")
            src = backgrounds[r.randrange(len(backgrounds))]
            # Chọn LẠI cho tới khi khác nguồn, thay vì bỏ qua im lặng.
            bg = src
            for _ in range(8):
                bg = backgrounds[r.randrange(len(backgrounds))]
                if bg != src:
                    break
            if bg == src:
                bg = backgrounds[(backgrounds.index(src) + 1) % len(backgrounds)]
            out_img = img_dir / f"{PREFIX}neg{j:05d}_{src.stem}.jpg"
            out_lbl = lbl_dir / f"{PREFIX}neg{j:05d}_{src.stem}.txt"
            if out_img.exists():
                continue
            if make_one(src, [], bg, out_img, out_lbl, r) is None:
                failed += 1
                continue
            created.append([rel(out_img), rel(out_lbl)])
            n_neg_made += 1

    total_tracked = save_manifest(created)

    print(f"\n[XONG] Đã sinh {len(created)} ảnh: "
          f"{len(created) - n_neg_made} dương ({boxes} hộp) + {n_neg_made} âm. "
          f"Lỗi {failed}.")
    print(f"       Manifest theo dõi {total_tracked} file: {MANIFEST}")
    print("\n   Bước tiếp theo:")
    print("     python audit_dataset.py --data data/weapon_8cls.yaml")
    print("     (kiểm lại: trung vị diện tích hộp phải GIẢM rõ rệt)")


def do_undo():
    if not MANIFEST.exists():
        print(f"[!] Không thấy manifest {MANIFEST}.")
        return
    created = load_manifest()
    n = 0
    for img_p, lbl_p in created:
        for p in (unrel(img_p), unrel(lbl_p)):
            if p.exists():
                p.unlink()
                n += 1
    MANIFEST.unlink()
    print(f"[XONG] Đã xoá {n} file do script sinh ra và xoá manifest.")


def main():
    ap = argparse.ArgumentParser(
        description="Thu nhỏ ảnh có hộp quá to, dán vào ảnh nền thật")
    ap.add_argument("--apply", action="store_true", help="thực hiện thật")
    ap.add_argument("--undo", action="store_true", help="xoá mọi ảnh đã sinh")
    ap.add_argument("--limit", type=int, default=None,
                    help="giới hạn số ảnh dương để thử nhanh")
    args = ap.parse_args()

    if not MERGED.is_dir():
        print(f"[!] Không thấy {MERGED}")
        sys.exit(1)

    if args.undo:
        do_undo()
    elif args.apply:
        do_apply(args.limit)
    else:
        oversized, backgrounds = scan()
        report(oversized, backgrounds, int(len(oversized) * NEG_RATIO))
        print("\n[!] Đây là bản xem trước, chưa ghi gì.")
        print("    Chạy lại với --apply để thực hiện.")


if __name__ == "__main__":
    main()
