"""
Offline Video Processor for Weapon Detection & Tracking.
Takes any input video, processes frame-by-frame with YOLO26s + ByteTrack,
draws prominent bounding circles/boxes with weapon labels, and exports a finished MP4 video.

Usage:
  python process_video.py
  python process_video.py "path/to/video.mp4"
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import warnings
warnings.filterwarnings("ignore")
from src.common.class_policy import load_policy, TIER_WEAPON
from src.common.detector import WeaponDetector

# Chính sách lớp dùng chung — xem khối class_policy trong system_config.yaml
POLICY = load_policy()


# GHI CHÚ: hai bảng VIETNAMESE_NAMES và COLORS trước đây nằm ở đây đã
# được chuyển vào khối `class_policy.display` trong
# configs/system_config.yaml. Giữ bảng riêng ở từng file chính là
# nguyên nhân khiến mô hình đổi từ 3 sang 12 lớp mà có file được sửa
# có file không. Dùng POLICY.vi_name() / POLICY.color() thay thế.


def draw_hud_annotation(frame, box, class_name, conf, track_id, is_alert=True):
    """
    Vẽ khoanh tròn / khung radar nổi bật và gắn nhãn tên hung khí rõ nét.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(w, h) // 2 + 10

    color = POLICY.color(class_name)
    vn_name = POLICY.vi_name(class_name)
    if POLICY.tier(class_name) != TIER_WEAPON:
        vn_name = f"[NV] {vn_name}"   # nghi vấn: dụng cụ, không phải vũ khí

    # 1. Vẽ VÒNG TRÒN bao quanh hung khí (khoanh tròn theo yêu cầu)
    cv2.circle(frame, (cx, cy), radius, color, 3, lineType=cv2.LINE_AA)
    # Tâm ngắm radar
    cv2.circle(frame, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

    # 2. Vẽ 4 góc radar bao ngoài
    corner_len = max(12, int(min(w, h) * 0.25))
    t = 2
    # Góc trên-trái
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
    # Góc trên-phải
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
    # Góc dưới-trái
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
    # Góc dưới-phải
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

    # 3. Gắn nhãn tên hung khí với nền nổi bật tương phản cao
    label_text = f" {vn_name} | {int(conf * 100)}% [ID:{track_id}] "
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    badge_x1 = max(0, cx - text_w // 2)
    badge_y1 = max(0, cy - radius - text_h - 14)
    badge_x2 = badge_x1 + text_w
    badge_y2 = badge_y1 + text_h + 8

    # Nền nhãn (Pill badge)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

    # Chữ màu trắng hoặc đen đậm
    text_color = (255, 255, 255)
    cv2.putText(frame, label_text, (badge_x1, badge_y2 - 4), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def process_video_file(
    input_path: str,
    output_path: str = None,
    weights_path: str = "weights/best.pt",
    imgsz: int = None,
    conf_thresh: float = None,
    device: str = None,
    weapons_only: bool = False
):
    """
    conf_thresh = None  -> dùng ngưỡng RIÊNG từng lớp (khuyến nghị).
    conf_thresh = số    -> ép một ngưỡng chung cho mọi lớp.

    imgsz / device = None -> đọc từ mục `inference` của system_config.yaml.
    """
    if not os.path.exists(input_path):
        print(f"[!] Lỗi: Không tìm thấy file video: {input_path}")
        return None

    # Hàm tra ngưỡng XÁC NHẬN theo lớp. Ngưỡng DÒ (thấp) do lõi phát hiện lo.
    if conf_thresh is None:
        def cls_thresholds(name):
            return POLICY.threshold(name)
    else:
        def cls_thresholds(name):
            return float(conf_thresh)

    # Khởi tạo đường dẫn output
    input_p = Path(input_path)
    if output_path is None:
        out_dir = Path("storage/processed_videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{input_p.stem}_detected.mp4")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎬 CÔNG CỤ XỬ LÝ & KHOANH VÙNG VŨ KHÍ TRONG VIDEO")
    print(f"Video đầu vào : {input_path}")
    print(f"Video đầu ra  : {output_path}")
    print(f"Mô hình AI    : {weights_path}")
    print(f"Thiết bị      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] Không thể mở video: {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Khởi tạo VideoWriter MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[+] Đang tải mô hình YOLO...")
    # BẢNG TÊN LỚP GÁN CỨNG 12 LỚP ĐÃ BỊ BỎ KHỎI ĐÂY - xem ghi chú trong
    # process_image.py. Tên lớp chỉ có một nguồn: chính file .pt.
    detector = WeaponDetector(
        weights=weights_path, imgsz=imgsz, device=device,
        conf_override=conf_thresh, weapons_only=weapons_only,
    )
    print(detector.describe())

    # TRƯỚC ĐÂY: WeaponByteTracker(track_high_thresh=0.45, track_low_thresh=0.15)
    # Gán cứng, và BỎ QUÊN new_track_thresh. Hậu quả: bản sửa
    # new_track_thresh=0.35 chỉ có tác dụng ở pipeline camera, còn đường quét
    # video này vẫn dùng mặc định của tracker. Nay đọc hết từ mục `tracking`
    # của system_config.yaml.
    tracker = detector.make_tracker()

    print(f"[+] Bắt đầu xử lý {total_frames} khung hình ({total_frames / fps:.1f} giây video)...")
    frame_idx = 0
    threats_count = 0
    geo_rejected = 0    # số hộp bị loại vì hình học bất thường
    t_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Nhận diện. detect_for_tracking() dò ở ngưỡng THẤP và ĐÃ áp lọc
        # hình học, nhưng CHƯA áp ngưỡng riêng từng lớp - để ByteTrack giai
        # đoạn 2 còn detection mờ mà bám vết. Xác nhận theo ngưỡng lớp làm ở
        # bước dưới, trên vết đã bám.
        detections, geo_rej = detector.detect_for_tracking(frame)
        geo_rejected += len(geo_rej)

        # 2. Bám vết bằng ByteTrack
        active_tracks = tracker.update(detections)

        # 3. Vẽ khoanh tròn và nhãn tên cho từng hung khí
        frame_threats = 0
        for track in active_tracks:
            if track.time_since_update > 0:
                continue

            box = track.get_current_box()
            cls_name = detector.class_name(track.cls_id)

            # --- Xác nhận theo ngưỡng RIÊNG của lớp --------------------
            # Trước đây dùng một cặp số cứng (0.35 / 0.55) cho cả 12 lớp.
            # Nay mỗi lớp có ngưỡng riêng: súng 0.30, dao 0.45, xích 0.60.
            # Bám vết nhiều frame vẫn được hạ ngưỡng một bậc, vì hiện diện
            # bền vững qua nhiều frame tự nó đã là bằng chứng.
            cls_thresh = cls_thresholds(cls_name)
            # weapons_only và lọc hình học ĐÃ làm ở detect_for_tracking(),
            # không lặp lại ở đây. Hộp bám vết có thể trôi khỏi hộp gốc, nên
            # kiểm hình học một lần nữa trên hộp đã lọc Kalman:
            ok_geo, _ = POLICY.keep_geometry(
                box, width, height, conf=track.conf, cls_name=cls_name)
            if not ok_geo:
                geo_rejected += 1
                continue
            is_confirmed = (track.hits >= 3 and track.conf >= cls_thresh * 0.8) \
                or track.conf >= cls_thresh

            if is_confirmed:
                frame_threats += 1
                threats_count = max(threats_count, track.id + 1)
                draw_hud_annotation(
                    frame=frame,
                    box=box,
                    class_name=cls_name,
                    conf=track.conf,
                    track_id=track.id,
                    is_alert=True
                )

        # 4. Vẽ thanh hiển thị trạng thái an ninh phía trên cùng (HUD Banner)
        time_sec = frame_idx / fps
        time_str = f"{int(time_sec // 60):02d}:{time_sec % 60:05.2f}"

        if frame_threats > 0:
            # 🚨 Viền đỏ báo động toàn bộ khung hình
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
            # Banner đỏ đậm cảnh báo khẩn cấp
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, 0), (width, 50), (0, 0, 180), -1)
            cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
            status_text = f"THOI GIAN: {time_str} | TONG MUC TIEU: {threats_count}"
            cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            alert_badge = f"!! CANH BAO DO: {frame_threats} HUNG KHI !!"
            cv2.putText(frame, alert_badge, (max(width - 450, 200), 32), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        else:
            # Banner đen trong suốt trên cùng
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (0, 0), (width, 50), (20, 20, 20), -1)
            cv2.addWeighted(banner_overlay, 0.65, frame, 0.35, 0, frame)
            status_text = f"THOI GIAN: {time_str} | TONG MUC TIEU: {threats_count}"
            cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            safe_badge = "AN TOAN"
            cv2.putText(frame, safe_badge, (width - 150, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        # Ghi khung hình vào video đầu ra
        writer.write(frame)

        # In tiến độ xử lý
        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = time.perf_counter() - t_start
            speed = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / speed if speed > 0 else 0
            bar = "█" * int(pct // 5) + "-" * (20 - int(pct // 5))
            sys.stdout.write(f"\r  [{bar}] {pct:.1f}% ({frame_idx}/{total_frames}) | {speed:.1f} FPS | Con lai: {eta:.0f}s")
            sys.stdout.flush()

    cap.release()
    writer.release()
    elapsed_total = time.perf_counter() - t_start

    print(f"\n\n[SUCCESS] Hoan thanh xu ly video trong {elapsed_total:.1f} giay!")
    if geo_rejected:
        print(f"[LOC HINH HOC] Da loai {geo_rejected} hop bat thuong "
              f"(qua to / qua nho / ti le canh la) - thuong la canh tay, balo, mep tuong.")
    print(f"[+] Video da duoc luu tai: {os.path.abspath(output_path)}")
    print("=" * 70)

    # Tự động mở thư mục chứa video trên Windows để người dùng xem ngay
    try:
        abs_out = os.path.abspath(output_path)
        os.system(f'explorer.exe /select,"{abs_out}"')
    except Exception:
        pass

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and draw circled weapons with labels")
    parser.add_argument("input", nargs="?", default=None, help="Duong dan toi file video can xu ly")
    parser.add_argument("--output", "-o", default=None, help="Duong dan video dau ra")
    parser.add_argument("--weights", "-w", default="weights/best.pt", help="Trong so model")
    parser.add_argument("--conf", type=float, default=None,
                        help="Ep MOT nguong chung cho moi lop. Bo trong (mac dinh) de dung "
                             "nguong RIENG tung lop trong configs/system_config.yaml")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Kich thuoc anh suy luan. Bo trong de doc tu "
                             "configs/system_config.yaml (inference.imgsz = 960)")
    parser.add_argument("--weapons-only", action="store_true",
                        help="Chi hien vu khi that, bo qua nhom dung cu")
    args = parser.parse_args()

    print(POLICY.summary())
    if args.conf is not None:
        print(f"[LUU Y] Dang ep nguong chung {args.conf} - bo qua nguong rieng tung lop.")

    input_file = args.input

    if input_file is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            print("[+] Vui long chon video can xu ly tu cua so Windows...")
            selected = filedialog.askopenfilename(
                title="Chon Video de xu ly khoanh vung vu khi",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"), ("All files", "*.*")]
            )
            root.destroy()
            if selected:
                input_file = selected
        except Exception:
            pass

    if not input_file:
        print("[!] Ban chua chon video nao. Hay chay: python process_video.py \"duong_dan_video.mp4\"")
        sys.exit(1)

    process_video_file(
        input_path=input_file,
        output_path=args.output,
        weights_path=args.weights,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        weapons_only=args.weapons_only
    )
