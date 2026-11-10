"""
swing_bridge.py - Cầu nối Python xử lý hung khí cho giao diện Java Swing.

Cung cấp các lệnh CLI và HTTP JSON cho ứng dụng Java Swing:
1. Xử lý ảnh:
   python swing_bridge.py --action image --input "path/to/input.jpg" --output "path/to/out.jpg"
2. Xử lý video:
   python swing_bridge.py --action video --input "path/to/input.mp4" --output "path/to/out.mp4"
3. Camera trực tiếp:
   python swing_bridge.py --action camera
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
import argparse
from pathlib import Path

# Đảm bảo đường dẫn gốc được nạp
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.common.detector import WeaponDetector


def process_image_cli(input_path, output_path, weights="weights/best.pt", imgsz=None):
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(json.dumps({"status": "error", "message": f"File không tồn tại: {input_path}"}, ensure_ascii=False))
        return

    # Đọc ảnh an toàn kể cả khi đường dẫn có dấu tiếng Việt trên Windows
    try:
        data = np.fromfile(input_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        frame = cv2.imread(input_path)

    if frame is None:
        print(json.dumps({"status": "error", "message": f"Không thể đọc file ảnh: {input_path}"}, ensure_ascii=False))
        return

    weights_path = os.path.abspath(weights) if not os.path.isabs(weights) else weights
    if not os.path.exists(weights_path):
        weights_path = str(PROJECT_ROOT / "weights" / "best.pt")

    detector = WeaponDetector(weights=weights_path, imgsz=imgsz)
    dets, rejected = detector.detect(frame)
    annotated = detector.draw(frame, dets)

    if output_path is None:
        out_dir = PROJECT_ROOT / "storage" / "processed_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{Path(input_path).stem}_detected.jpg")

    try:
        is_ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if is_ok:
            buf.tofile(output_path)
        else:
            cv2.imwrite(output_path, annotated)
    except Exception:
        cv2.imwrite(output_path, annotated)

    n_weapon = sum(1 for d in dets if d.is_weapon)
    items = []
    for d in dets:
        items.append({
            "class": d.cls_name,
            "vi_name": d.vi_name,
            "conf": round(d.conf, 2),
            "is_weapon": d.is_weapon,
            "box": d.box
        })

    rejected_items = []
    for name, conf, why in rejected:
        rejected_items.append({
            "class": name,
            "conf": round(float(conf), 2),
            "reason": str(why)
        })

    result = {
        "status": "success",
        "output_image": os.path.abspath(output_path),
        "total_detections": len(dets),
        "weapon_count": n_weapon,
        "is_alert": n_weapon > 0,
        "items": items,
        "rejected": rejected_items
    }
    print(json.dumps(result, ensure_ascii=False))


def process_video_cli(input_path, output_path, weights="weights/best.pt", imgsz=None):
    from process_video import process_video_file
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(json.dumps({"status": "error", "message": f"File video không tồn tại: {input_path}"}, ensure_ascii=False))
        return

    if output_path is None:
        out_dir = PROJECT_ROOT / "storage" / "processed_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{Path(input_path).stem}_detected.mp4")

    weights_path = os.path.abspath(weights) if not os.path.isabs(weights) else weights
    if not os.path.exists(weights_path):
        weights_path = str(PROJECT_ROOT / "weights" / "best.pt")

    try:
        out_file = process_video_file(
            input_path=input_path,
            output_path=output_path,
            weights_path=weights_path,
            imgsz=imgsz
        )

        result = {
            "status": "success",
            "output_video": os.path.abspath(out_file),
        }
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False))


def stream_live_camera(weights="weights/best.pt", imgsz=640, cam_index=0, source=None):
    """
    Truyền luồng camera trực tiếp (hoặc video mô phỏng CCTV) vào Java Swing.
    Được truyền qua stdout theo chuẩn giao thức nhị phân:
    [4 bytes Big-Endian: độ dài JSON Header]
    [JSON bytes (UTF-8)]
    [4 bytes Big-Endian: độ dài ảnh JPEG]
    [JPEG bytes]

    Tương thích 100% với môi trường không có GUI OpenCV (opencv-python-headless).
    Giao diện hiển thị trực tiếp và mượt mà trong Java Swing JPanel!
    """
    import struct
    # Giữ binary stdout, chuyển text stdout sang stderr để tránh log/warning làm hỏng dữ liệu nhị phân
    raw_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    detector = WeaponDetector(weights=weights, imgsz=imgsz)

    cap = None
    if source and os.path.exists(source):
        cap = cv2.VideoCapture(source)
    else:
        # Thử mở webcam với backend DSHOW rồi backend mặc định
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened() and cam_index == 0:
            cap = cv2.VideoCapture(1)

    if cap is None or not cap.isOpened():
        err_meta = json.dumps({
            "status": "error",
            "message": f"Không thể kết nối tới Camera (index {cam_index}). Vui lòng kiểm tra quyền webcam trong Windows Settings hoặc chọn video mô phỏng!"
        }, ensure_ascii=False).encode('utf-8')
        raw_out.write(struct.pack('>I', len(err_meta)))
        raw_out.write(err_meta)
        raw_out.write(struct.pack('>I', 0))
        raw_out.flush()
        return

    is_video_file = bool(source and os.path.exists(source))
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        curr_time = time.time()
        fps = 1.0 / max(1e-5, (curr_time - prev_time))
        prev_time = curr_time

        dets, _ = detector.detect(frame)
        canvas = detector.draw(frame, dets)

        n_weapon = sum(1 for d in dets if d.is_weapon)
        items = [{
            "class": d.cls_name,
            "vi_name": d.vi_name,
            "conf": round(d.conf, 2),
            "is_weapon": d.is_weapon
        } for d in dets]

        meta = json.dumps({
            "status": "ok",
            "fps": round(fps, 1),
            "is_alert": n_weapon > 0,
            "weapon_count": n_weapon,
            "items": items
        }, ensure_ascii=False).encode('utf-8')

        ret_enc, buffer = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret_enc:
            continue

        img_bytes = buffer.tobytes()

        try:
            raw_out.write(struct.pack('>I', len(meta)))
            raw_out.write(meta)
            raw_out.write(struct.pack('>I', len(img_bytes)))
            raw_out.write(img_bytes)
            raw_out.flush()
        except (BrokenPipeError, IOError):
            break

    cap.release()


def run_live_camera_window(weights="weights/best.pt", imgsz=640, cam_index=0):
    """Mở cửa sổ OpenCV bên ngoài (nếu môi trường có hỗ trợ OpenCV HighGUI)."""
    detector = WeaponDetector(weights=weights, imgsz=imgsz)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(json.dumps({"status": "error", "message": "Không thể mở camera"}))
        return

    try:
        cv2.namedWindow("CAMERA GIAM SAT HUNG KHI (JAVA SWING BRIDGE)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CAMERA GIAM SAT HUNG KHI (JAVA SWING BRIDGE)", 960, 540)
    except cv2.error as e:
        print(json.dumps({"status": "error", "message": f"OpenCV HighGUI không khả dụng: {e}. Vui lòng dùng chế độ stream cho Java Swing!"}))
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets, _ = detector.detect(frame)
        canvas = detector.draw(frame, dets)

        cv2.imshow("CAMERA GIAM SAT HUNG KHI (JAVA SWING BRIDGE)", canvas)
        if cv2.waitKey(1) & 0xFF == 27:  # Phím ESC để đóng
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swing Bridge CLI")
    parser.add_argument("--action", choices=["image", "video", "camera", "camera_window"], required=True)
    parser.add_argument("--input", "-i", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--weights", "-w", default="weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    # Mặc định imgsz thông minh theo từng action nếu không chỉ định rõ:
    # - Ảnh và video offline: 960 (khớp kích thước train/fine-tune, tối đa hóa recall)
    # - Camera trực tiếp: 640 (duy trì FPS realtime)
    if args.imgsz is not None:
        resolved_imgsz = args.imgsz
    elif args.action in ("image", "video"):
        resolved_imgsz = 960
    else:
        resolved_imgsz = 640

    if args.action == "image":
        process_image_cli(args.input, args.output, weights=args.weights, imgsz=resolved_imgsz)
    elif args.action == "video":
        process_video_cli(args.input, args.output, weights=args.weights, imgsz=resolved_imgsz)
    elif args.action == "camera":
        stream_live_camera(weights=args.weights, imgsz=resolved_imgsz, cam_index=args.cam, source=args.input)
    elif args.action == "camera_window":
        run_live_camera_window(weights=args.weights, imgsz=resolved_imgsz, cam_index=args.cam)

