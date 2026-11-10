"""
collect_hard_negatives.py - Dạy mô hình "cánh tay không phải vũ khí".

VẤN ĐỀ
------
Mô hình vẽ hộp lên cánh tay, balo, mép tường. Nguyên nhân KHÔNG phải
ngưỡng, mà là thiếu dữ liệu âm: bộ Roboflow toàn ảnh chụp vật thể làm
chủ thể, mô hình chưa từng thấy một khung hình đời thật có người đeo
balo và được dạy rằng ở đó KHÔNG CÓ GÌ.

CÁCH SỬA RẺ NHẤT TRONG HỌC MÁY
------------------------------
Hard negative mining. Bạn đưa vào video KHÔNG CÓ VŨ KHÍ, script chạy mô
hình trên đó, và MỌI detection đều chắc chắn là sai. Lưu những khung đó
kèm file nhãn RỖNG, trộn vào tập train, fine-tune lại.

Điểm quan trọng: **không cần gán nhãn thủ công một hộp nào**. Nhãn rỗng
nghĩa là "khung này không có vũ khí" - và đó chính xác là điều mô hình
đang cần học. Chi phí gần như bằng không so với việc gán nhãn vật thể.

QUY TRÌNH 3 BƯỚC
----------------
  B1. Quay / tải vài video KHÔNG có vũ khí, giống môi trường triển khai:
      hành lang, lớp học, người đi lại đeo balo, cầm điện thoại, ô...

  B2. Thu thập:
        python collect_hard_negatives.py --input clip1.mp4 clip2.mp4
      hoặc bỏ trống để mở hộp thoại chọn nhiều file.

  B3. Trộn vào tập train rồi fine-tune:
        python collect_hard_negatives.py --merge
        python src/training/train.py --weights weights/best.pt --epochs 15 --lr0 0.0005

CẢNH BÁO AN TOÀN
----------------
Chỉ đưa vào video THẬT SỰ KHÔNG CÓ vũ khí. Nếu video có vũ khí mà bạn
gán nhãn rỗng, bạn đang dạy mô hình BỎ QUA vũ khí - hậu quả ngược hẳn
điều mong muốn. Script luôn lưu kèm ảnh xem trước có vẽ hộp để bạn kiểm
tra lại trước khi trộn.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import warnings
warnings.filterwarnings("ignore")
import torch
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

NEG_ROOT = Path("data/hard_negatives")
IMG_DIR = NEG_ROOT / "images"
LBL_DIR = NEG_ROOT / "labels"
PREVIEW_DIR = NEG_ROOT / "preview"
PROBE_CONF = 0.10          # dò thấp để bắt được cả nhiễu yếu


def pick_videos_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print("[+] Chọn MỘT HOẶC NHIỀU video KHÔNG CÓ VŨ KHÍ (giữ Ctrl để chọn nhiều)...")
        files = filedialog.askopenfilenames(
            title="Chọn video KHÔNG có vũ khí để thu thập hard negative",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                       ("All files", "*.*")],
        )
        root.destroy()
        return list(files)
    except Exception as e:
        print(f"[!] Không mở được hộp thoại ({e}). Hãy dùng --input đường/dẫn.mp4")
        return []


def collect(videos, weights, imgsz, max_per_video, stride, device):
    for d in (IMG_DIR, LBL_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    grand_total = 0

    for vpath in videos:
        if not os.path.exists(vpath):
            print(f"[!] Bỏ qua, không tìm thấy: {vpath}")
            continue

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[!] Bỏ qua, không mở được: {vpath}")
            continue

        stem = Path(vpath).stem.replace(" ", "_")[:40]
        print(f"\n[+] Quét: {vpath}")
        idx = 0
        saved = 0
        by_class = {}

        while cap.isOpened() and saved < max_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % stride != 0:
                continue

            r = model.predict(frame, imgsz=imgsz, conf=PROBE_CONF,
                              device=device, verbose=False)
            if not r or r[0].boxes is None or len(r[0].boxes) == 0:
                continue

            # Có detection trên video không vũ khí -> toàn bộ là FALSE POSITIVE
            saved += 1
            grand_total += 1
            name = f"neg_{stem}_{saved:05d}"

            cv2.imwrite(str(IMG_DIR / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            # Nhãn RỖNG = "khung này không có vũ khí"
            open(LBL_DIR / f"{name}.txt", "w", encoding="utf-8").close()

            # Ảnh xem trước có vẽ hộp, để người dùng kiểm tra lại
            prev = frame.copy()
            for b in r[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                cid = int(b.cls[0].cpu().numpy())
                cf = float(b.conf[0].cpu().numpy())
                cn = names.get(cid, str(cid))
                by_class[cn] = by_class.get(cn, 0) + 1
                cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(prev, f"{cn} {cf:.2f}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(PREVIEW_DIR / f"{name}.jpg"), prev,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])

        cap.release()
        print(f"    Lưu {saved} khung có báo nhầm.")
        if by_class:
            top = sorted(by_class.items(), key=lambda x: -x[1])
            print("    Lớp bị nhầm nhiều nhất: " +
                  ", ".join(f"{k}={v}" for k, v in top[:5]))

    print("\n" + "=" * 72)
    print(f"   ĐÃ THU THẬP {grand_total} HARD NEGATIVE")
    print("=" * 72)
    print(f"   Ảnh      : {IMG_DIR}")
    print(f"   Nhãn rỗng: {LBL_DIR}")
    print(f"   Xem trước: {PREVIEW_DIR}   <-- MỞ THƯ MỤC NÀY KIỂM TRA TRƯỚC")
    print()
    print("   Mở thư mục xem trước. Nếu khung nào THẬT SỰ có vũ khí, hãy xoá")
    print("   cả ảnh lẫn nhãn tương ứng - giữ lại sẽ dạy mô hình bỏ qua vũ khí.")
    print()
    print("   Kiểm tra xong, trộn vào tập train:")
    print("     python collect_hard_negatives.py --merge")
    print("=" * 72)
    return grand_total


def merge_into_train(train_img="train/images", train_lbl="train/labels"):
    if not IMG_DIR.exists():
        print(f"[!] Chưa có {IMG_DIR}. Chạy thu thập trước.")
        return 0

    ti, tl = Path(train_img), Path(train_lbl)
    if not ti.exists() or not tl.exists():
        print(f"[!] Không tìm thấy {ti} hoặc {tl}.")
        return 0

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if not imgs:
        print("[!] Không có ảnh nào để trộn.")
        return 0

    n_train = len(list(ti.glob("*.*")))
    print(f"[+] Tập train hiện có   : {n_train} ảnh")
    print(f"[+] Hard negative sẽ trộn: {len(imgs)} ảnh "
          f"({len(imgs) / max(1, n_train + len(imgs)) * 100:.1f}% tập mới)")

    if len(imgs) > n_train * 0.3:
        print("[!] CẢNH BÁO: ảnh âm chiếm hơn 30% tập train.")
        print("    Quá nhiều ảnh âm sẽ kéo recall xuống. Nên giữ khoảng 10-20%.")

    ans = input("\n    Xác nhận trộn? (y/n): ").strip().lower()
    if ans != "y":
        print("    Đã huỷ.")
        return 0

    moved = 0
    for img in imgs:
        lbl = LBL_DIR / f"{img.stem}.txt"
        shutil.copy2(img, ti / img.name)
        if lbl.exists():
            shutil.copy2(lbl, tl / lbl.name)
        else:
            open(tl / f"{img.stem}.txt", "w", encoding="utf-8").close()
        moved += 1

    # Xoá cache cũ, nếu không ultralytics đọc nhầm danh sách ảnh cũ
    for c in Path(".").glob("*/labels.cache"):
        try:
            c.unlink()
            print(f"[+] Đã xoá cache cũ: {c}")
        except Exception:
            pass

    print(f"\n[XONG] Đã trộn {moved} ảnh âm vào tập train.")
    print("\n   Fine-tune từ trọng số hiện tại (KHÔNG train lại từ đầu):")
    print("     python src/training/train.py --weights weights/best.pt \\")
    print("            --epochs 15 --lr0 0.0005 --name weapon_hardneg")
    print("\n   Learning rate thấp (0.0005) để mô hình học 'đây không phải vũ khí'")
    print("   mà không quên những gì đã học về vũ khí thật.")
    return moved


def main():
    ap = argparse.ArgumentParser(
        description="Thu thập hard negative từ video KHÔNG CÓ vũ khí")
    ap.add_argument("--input", "-i", nargs="*", default=None,
                    help="Một hoặc nhiều video. Bỏ trống để mở hộp thoại chọn file.")
    ap.add_argument("--weights", "-w", default="weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-per-video", type=int, default=150,
                    help="Số khung tối đa lấy từ mỗi video (mặc định 150)")
    ap.add_argument("--stride", type=int, default=10,
                    help="Lấy 1 khung mỗi N khung (mặc định 10, tránh ảnh trùng nhau)")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge", action="store_true",
                    help="Trộn hard negative đã thu thập vào tập train")
    args = ap.parse_args()

    if args.merge:
        merge_into_train()
        return

    videos = args.input if args.input else pick_videos_dialog()
    if not videos:
        print("[!] Chưa chọn video nào. Kết thúc.")
        return

    print("=" * 72)
    print("   THU THẬP HARD NEGATIVE")
    print("   CHỈ dùng video KHÔNG CÓ vũ khí. Mọi detection tìm được đều")
    print("   được coi là BÁO NHẦM và sẽ thành dữ liệu âm.")
    print("=" * 72)
    for v in videos:
        print(f"   - {v}")

    collect(videos, args.weights, args.imgsz,
            args.max_per_video, args.stride, args.device)


if __name__ == "__main__":
    main()
