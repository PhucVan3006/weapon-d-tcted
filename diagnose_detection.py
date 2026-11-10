"""
diagnose_detection.py - Đo xem mô hình THẬT SỰ nhìn thấy gì trên video CỦA BẠN.

VÌ SAO CẦN FILE NÀY
-------------------
Ngưỡng trong configs/system_config.yaml được suy ra từ ma trận nhầm lẫn
trên TẬP TEST - tức dữ liệu cùng nguồn với tập train. Video giám sát thật
có phân bố độ tin cậy khác hẳn: mờ hơn, nén nặng hơn, vật thể nhỏ hơn.
Áp ngưỡng của tập test lên video thật là đoán, không phải đo.

Script này chạy mô hình ở ngưỡng rất thấp (0.05) trên video của bạn rồi
báo cáo:
  1. Phân bố độ tin cậy của TỪNG LỚP (trung vị, các phân vị)
  2. Kích thước vật thể so với khung hình - cho biết có phải đang bỏ sót
     vì vật quá nhỏ không
  3. So sánh nhiều mức imgsz để thấy ảnh hưởng của độ phân giải
  4. Với mỗi ngưỡng ứng viên, còn lại bao nhiêu detection

KHÔNG tự sửa config. Nó chỉ đưa số liệu để BẠN quyết định.

CÁCH DÙNG
---------
    python diagnose_detection.py video.mp4
    python diagnose_detection.py video.mp4 --imgsz 640 960 1280
    python diagnose_detection.py video.mp4 --sample 300
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import warnings
warnings.filterwarnings("ignore")
import torch
from ultralytics import YOLO

from src.common.class_policy import load_policy, TIER_WEAPON

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

POLICY = load_policy()
PROBE_CONF = 0.05          # ngưỡng dò rất thấp để thấy TOÀN BỘ ứng viên
CANDIDATE_THRESHOLDS = [0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def probe(video_path, weights, imgsz, sample, device):
    """Chạy mô hình trên các khung lấy mẫu, thu mọi detection từ conf 0.05."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Không mở được video: {video_path}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(w * h) if w and h else 1.0

    stride = max(1, total // sample) if total > 0 else 1
    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    confs = defaultdict(list)      # lớp -> danh sách conf
    areas = defaultdict(list)      # lớp -> % diện tích khung
    frames_done = 0
    t0 = time.perf_counter()
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % stride != 0:
            continue
        frames_done += 1
        if frames_done > sample:
            break

        r = model.predict(frame, imgsz=imgsz, conf=PROBE_CONF,
                          device=device, verbose=False)
        if r and r[0].boxes is not None:
            for b in r[0].boxes:
                cid = int(b.cls[0].cpu().numpy())
                cf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
                name = names.get(cid, f"class_{cid}")
                confs[name].append(cf)
                areas[name].append((x2 - x1) * (y2 - y1) / frame_area * 100.0)

    cap.release()
    elapsed = time.perf_counter() - t0
    return {
        "imgsz": imgsz, "confs": confs, "areas": areas,
        "frames": frames_done, "seconds": elapsed,
        "video": {"w": w, "h": h, "fps": fps, "total": total},
    }


def report(res):
    imgsz = res["imgsz"]
    confs, areas = res["confs"], res["areas"]
    n_det = sum(len(v) for v in confs.values())
    speed = res["frames"] / res["seconds"] if res["seconds"] > 0 else 0

    print("\n" + "=" * 92)
    print(f"   imgsz = {imgsz}   |   {res['frames']} khung đã quét   |   "
          f"{n_det} detection thô (conf >= {PROBE_CONF})   |   {speed:.1f} FPS")
    print("=" * 92)

    if n_det == 0:
        print("   KHÔNG có detection nào, kể cả ở ngưỡng 0.05.")
        print("   -> Mô hình hoàn toàn không nhận ra gì trên video này. Đây là")
        print("      khoảng cách miền dữ liệu, KHÔNG phải vấn đề ngưỡng.")
        return

    print(f"{'Lớp':<12}{'số det':>8}{'conf TV':>9}{'p25':>7}{'p75':>7}{'p90':>7}"
          f"{'% khung':>9}  {'mức'}")
    print("-" * 92)
    for name in sorted(confs, key=lambda k: -len(confs[k])):
        c = sorted(confs[name])
        a = sorted(areas[name])
        tier = "VŨ KHÍ" if POLICY.tier(name) == TIER_WEAPON else "nghi vấn"
        print(f"{name:<12}{len(c):>8}{percentile(c,0.5):>9.3f}"
              f"{percentile(c,0.25):>7.3f}{percentile(c,0.75):>7.3f}"
              f"{percentile(c,0.90):>7.3f}{percentile(a,0.5):>9.2f}  {tier}")
    print("-" * 92)

    # Còn lại bao nhiêu nếu áp từng ngưỡng
    print("\n   SỐ DETECTION CÒN LẠI THEO NGƯỠNG:")
    header = "   " + "lớp".ljust(12) + "".join(f"{t:>7.2f}" for t in CANDIDATE_THRESHOLDS)
    print(header)
    print("   " + "-" * (12 + 7 * len(CANDIDATE_THRESHOLDS)))
    for name in sorted(confs, key=lambda k: -len(confs[k])):
        c = confs[name]
        row = "   " + name.ljust(12)
        for t in CANDIDATE_THRESHOLDS:
            row += f"{sum(1 for v in c if v >= t):>7}"
        print(row)

    # Đối chiếu với ngưỡng đang áp dụng
    print("\n   NGƯỠNG ĐANG ÁP DỤNG so với dữ liệu video này:")
    for name in sorted(confs, key=lambda k: -len(confs[k])):
        c = sorted(confs[name])
        th = POLICY.threshold(name)
        kept = sum(1 for v in c if v >= th)
        pct = kept / len(c) * 100
        med = percentile(c, 0.5)
        flag = ""
        if pct < 20:
            flag = "  <-- NGƯỠNG QUÁ CAO, đang cắt gần hết"
        elif med < th:
            flag = "  <-- trung vị thấp hơn ngưỡng, nên cân nhắc hạ"
        print(f"     {name:<12} ngưỡng {th:.2f}  giữ lại {kept}/{len(c)} ({pct:.0f}%){flag}")

    # Cảnh báo vật thể nhỏ
    small = [n for n in areas if percentile(sorted(areas[n]), 0.5) < 0.5]
    if small:
        print(f"\n   [LƯU Ý] Các lớp sau có vật thể rất nhỏ (<0,5% diện tích khung): "
              f"{', '.join(small)}")
        print("           Đây là dấu hiệu nên TĂNG imgsz khi suy luận.")


def pick_video_dialog():
    """Mở hộp thoại chọn file của Windows. Trả về đường dẫn hoặc None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print("[+] Vui lòng chọn video cần chẩn đoán từ cửa sổ Windows...")
        selected = filedialog.askopenfilename(
            title="Chọn video để chẩn đoán phát hiện hung khí",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if selected:
            print(f"[+] Đã chọn: {selected}")
        return selected or None
    except Exception as e:
        print(f"[!] Không mở được hộp thoại chọn file ({e}).")
        print("    Hãy truyền thẳng đường dẫn: python diagnose_detection.py \"video.mp4\"")
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Chẩn đoán xem mô hình thật sự thấy gì trên video của bạn")
    ap.add_argument("input", nargs="?", default=None,
                    help="Đường dẫn video. Bỏ trống để mở hộp thoại chọn file.")
    ap.add_argument("--weights", "-w", default="weights/best.pt")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280],
                    help="Các mức imgsz cần so sánh (mặc định: 640 1280)")
    ap.add_argument("--sample", type=int, default=200,
                    help="Số khung lấy mẫu đều trên toàn video (mặc định 200)")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    video_path = args.input

    # Không truyền đường dẫn -> mở hộp thoại chọn file của Windows
    if video_path is None:
        video_path = pick_video_dialog()
        if not video_path:
            print("[!] Bạn chưa chọn video nào. Kết thúc.")
            return

    if not os.path.exists(video_path):
        print(f"[!] Không tìm thấy video: {video_path}")
        return

    print("=" * 92)
    print("   CHẨN ĐOÁN PHÁT HIỆN TRÊN VIDEO THẬT")
    print(f"   Video    : {video_path}")
    print(f"   Trọng số : {args.weights}")
    print(f"   Dò ở ngưỡng rất thấp {PROBE_CONF} để thấy TOÀN BỘ ứng viên")
    print("=" * 92)

    results = []
    for sz in args.imgsz:
        r = probe(video_path, args.weights, sz, args.sample, args.device)
        if r:
            results.append(r)
            report(r)

    if len(results) >= 2:
        print("\n" + "=" * 92)
        print("   SO SÁNH CÁC MỨC IMGSZ")
        print("=" * 92)
        base = results[0]
        n0 = sum(len(v) for v in base["confs"].values())
        print(f"{'imgsz':<10}{'số det':>10}{'so với mốc':>14}{'FPS':>9}")
        print("-" * 92)
        for r in results:
            n = sum(len(v) for v in r["confs"].values())
            delta = f"{(n - n0) / n0 * 100:+.0f}%" if n0 else "-"
            sp = r["frames"] / r["seconds"] if r["seconds"] > 0 else 0
            print(f"{r['imgsz']:<10}{n:>10}{delta:>14}{sp:>9.1f}")
        print("-" * 92)
        print("   Nếu imgsz lớn cho NHIỀU detection hơn đáng kể, video của bạn có")
        print("   vật thể nhỏ và bạn đang bỏ sót vì độ phân giải, không phải vì ngưỡng.")

    print("\n" + "=" * 92)
    print("   BƯỚC TIẾP THEO")
    print("=" * 92)
    print("   1. Lớp nào có cột 'giữ lại' dưới 20% -> ngưỡng đang quá cao, hạ xuống")
    print("      quanh mức trung vị (cột conf TV) trong configs/system_config.yaml")
    print("   2. Nếu imgsz lớn thắng rõ -> chạy suy luận ở mức đó:")
    print("        python process_video.py video.mp4 --imgsz 1280")
    print("   3. Nếu KHÔNG lớp nào có detection -> vấn đề là khoảng cách miền dữ")
    print("      liệu, phải gán nhãn vài trăm khung từ video thật rồi fine-tune")
    print("=" * 92)


if __name__ == "__main__":
    main()
