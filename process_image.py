"""
Offline Image Processor for Weapon Detection.
Takes an image (or directory of images), detects weapons using YOLO26s (weights/best.pt),
draws prominent circles/bounding boxes with Vietnamese weapon labels, and exports the annotated image.

Usage:
  python process_image.py
  python process_image.py "path/to/image.jpg"
  python process_image.py "path/to/image_folder"
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
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import warnings
warnings.filterwarnings("ignore")
from src.bestshot.laplacian_scorer import calculate_quality_score
from src.common.class_policy import load_policy, TIER_WEAPON
from src.common.detector import WeaponDetector

# Chính sách lớp dùng chung: tên hiển thị, màu, phân mức nguy hiểm và
# ngưỡng riêng từng lớp. Xem khối class_policy trong configs/system_config.yaml.
POLICY = load_policy()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# GHI CHÚ: hai bảng VIETNAMESE_NAMES và COLORS trước đây nằm ở đây đã
# được chuyển vào khối `class_policy.display` trong
# configs/system_config.yaml. Giữ bảng riêng ở từng file chính là
# nguyên nhân khiến mô hình đổi từ 3 sang 12 lớp mà có file được sửa
# có file không. Dùng POLICY.vi_name() / POLICY.color() thay thế.


def draw_weapon_annotation(image, box, class_name, conf, index=0):
    """
    Khoanh tròn hung khí và gắn huy hiệu nhãn tên nổi bật với độ tương phản cao.
    Vật ở mức "nghi vấn" (dụng cụ) được vẽ nét mảnh hơn và có tiền tố [NV]
    để phân biệt rõ với vũ khí thật.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(w, h) // 2 + 10

    color = POLICY.color(class_name)
    is_weapon = POLICY.tier(class_name) == TIER_WEAPON
    vn_name = POLICY.vi_name(class_name)
    if not is_weapon:
        vn_name = f"[NV] {vn_name}"

    # Vũ khí thật vẽ đậm, vật nghi vấn vẽ mảnh
    ring_t = 3 if is_weapon else 2

    # 1. Khoanh tròn mục tiêu kèm tâm ngắm radar
    cv2.circle(image, (cx, cy), radius, color, ring_t, lineType=cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

    # 2. Khung viền radar 4 góc
    corner_len = max(12, int(min(w, h) * 0.25))
    t = 2 if is_weapon else 1
    cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, t)
    cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, t)
    cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, t)
    cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, t)
    cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, t)
    cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, t)
    cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, t)
    cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, t)

    # 3. Gắn nhãn tên hung khí
    label_text = f" {vn_name} | {int(conf * 100)}% "
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    badge_x1 = max(0, cx - text_w // 2)
    badge_y1 = max(0, cy - radius - text_h - 14)
    badge_x2 = badge_x1 + text_w
    badge_y2 = badge_y1 + text_h + 8

    # Nền nhãn (Pill badge)
    cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

    # Chữ trắng rõ nét
    cv2.putText(image, label_text, (badge_x1, badge_y2 - 4), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def process_single_image(
    image_path: str,
    output_dir: str = "storage/processed_images",
    detector=None,
    weights_path: str = "weights/best.pt",
    imgsz: int = None,
    conf_thresh: float = None,
    device: str = None,
    save_crops: bool = True,
    augment: bool = None,
    weapons_only: bool = False
):
    """
    conf_thresh = None  -> dùng ngưỡng RIÊNG từng lớp (khuyến nghị).
    conf_thresh = số    -> ép một ngưỡng chung cho mọi lớp, bỏ qua chính sách.

    imgsz / augment / device = None -> đọc từ mục `inference` của
    configs/system_config.yaml. Trước đây imgsz gán cứng 640 trong khi mô
    hình train ở 960; lệch kích thước giữa train và suy luận là nguyên nhân
    sót vật nhỏ rất thường gặp.

    MỌI quyết định "có phải vũ khí không, tên gì" nay do WeaponDetector lo.
    File này chỉ còn đọc ảnh, đo độ nét, vẽ và ghi kết quả.
    """
    if not os.path.exists(image_path):
        print(f"[!] Không tìm thấy ảnh: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Không thể đọc định dạng ảnh: {image_path}")
        return None

    # BẢNG TÊN LỚP GÁN CỨNG 12 LỚP ĐÃ BỊ BỎ KHỎI ĐÂY.
    # Nó là bản sao thứ hai của sự thật: mô hình nay có 8 lớp, bảng cũ vẫn
    # ghi 12, và lớp nào lệch id thì hiện ra sai tên. WeaponDetector đọc tên
    # trực tiếp từ file .pt và báo lỗi nếu đọc không được.
    if detector is None:
        detector = WeaponDetector(
            weights=weights_path, imgsz=imgsz, device=device,
            conf_override=conf_thresh, weapons_only=weapons_only,
            augment=augment,
        )
    annotated = img.copy()
    base_name = Path(image_path).stem
    out_dir_p = Path(output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir_p / f"{base_name}_detected.jpg")

    # Lõi phát hiện lo cả bốn tầng lọc: ngưỡng sàn, ngưỡng riêng từng lớp,
    # hình học theo độ tin cậy, phân mức vũ khí / nghi vấn.
    t0 = time.perf_counter()
    dets, rejected = detector.detect(img)
    infer_time = (time.perf_counter() - t0) * 1000

    detections = []
    crops_dir = out_dir_p / "crops"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, d in enumerate(dets):
        xyxy = d.box
        # Đo độ nét Laplacian và lưu ảnh crop cận cảnh
        q_score, lap_var, roi = calculate_quality_score(img, xyxy, d.conf)
        crop_path = None
        if save_crops and roi.size > 0:
            crop_path = str(crops_dir / f"{base_name}_det{idx+1}_{d.cls_name}.jpg")
            cv2.imwrite(crop_path, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])

        draw_weapon_annotation(annotated, xyxy, d.cls_name, d.conf, index=idx)
        row = d.as_dict()
        row["laplacian_var"] = round(lap_var, 1)
        row["crop_path"] = crop_path
        detections.append(row)

    # Banner HUD hiển thị cảnh báo đỏ nổi bật khi phát hiện hung khí
    h, w = img.shape[:2]
    n_weapon = sum(1 for d in dets if d.is_weapon)
    if n_weapon > 0:
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        banner = annotated.copy()
        cv2.rectangle(banner, (0, 0), (w, 50), (0, 0, 180), -1)
        cv2.addWeighted(banner, 0.85, annotated, 0.15, 0, annotated)
        status_str = f"!! CANH BAO DO: PHAT HIEN {n_weapon} HUNG KHI NGUY HIEM !!"
        cv2.putText(annotated, status_str, (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    elif detections:
        banner = annotated.copy()
        cv2.rectangle(banner, (0, 0), (w, 45), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.7, annotated, 0.3, 0, annotated)
        status_str = f"THEO DOI: {len(detections)} vat nghi van (dung cu) | {infer_time:.1f}ms"
        cv2.putText(annotated, status_str, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 215, 255), 1, lineType=cv2.LINE_AA)
    else:
        banner = annotated.copy()
        cv2.rectangle(banner, (0, 0), (w, 45), (20, 20, 20), -1)
        cv2.addWeighted(banner, 0.7, annotated, 0.3, 0, annotated)
        status_str = f"AN TOAN - Khong phat hien hung khi | {infer_time:.1f}ms"
        cv2.putText(annotated, status_str, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 0), 1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Danh sách bị loại gồm CẢ BA lý do: dưới ngưỡng lớp, hình học, phân mức.
    # Nhãn cũ ghi "LỌC HÌNH HỌC" cho tất cả là sai sau khi gộp lõi.
    if rejected:
        print(f"[ĐÃ LOẠI] {len(rejected)} hộp không qua được bộ lọc:")
        for cn, cf, why in rejected[:6]:
            print(f"    - {cn} ({cf*100:.0f}%): {why}")
        if len(rejected) > 6:
            print(f"    ... và {len(rejected) - 6} hộp khác")

    return out_path, detections, infer_time


def main():
    parser = argparse.ArgumentParser(description="Process image and circle detected weapons with labels")
    parser.add_argument("input", nargs="?", default=None, help="Duong dan toi file anh hoac thu muc anh")
    parser.add_argument("--output", "-o", default="storage/processed_images", help="Thu muc luu ket qua")
    parser.add_argument("--weights", "-w", default="weights/best.pt", help="Trong so model YOLO")
    parser.add_argument("--conf", type=float, default=None,
                        help="Ep MOT nguong chung cho moi lop. Bo trong (mac dinh) de "
                             "dung nguong RIENG tung lop trong configs/system_config.yaml")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Kich thuoc anh suy luan. Bo trong de doc tu "
                             "configs/system_config.yaml (inference.imgsz = 960, "
                             "khop voi imgsz luc train)")
    parser.add_argument("--tta", action="store_true",
                        help="Bat Test-Time Augmentation: +1-2 mAP, cham hon ~3 lan")
    parser.add_argument("--weapons-only", action="store_true",
                        help="Chi hien vu khi that, bo qua nhom dung cu (Crowbar/Stick/Hammer/Spanner/Chain)")
    args = parser.parse_args()

    print(POLICY.summary())
    if args.conf is not None:
        print(f"[LUU Y] Dang ep nguong chung {args.conf} - bo qua nguong rieng tung lop.")

    input_path = args.input

    # Nếu không truyền đường dẫn, mở hộp thoại chọn file của Windows
    if input_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            print("[+] Vui long chon file anh can xu ly tu cua so Windows...")
            selected = filedialog.askopenfilename(
                title="Chon file anh de nhan dien va khoanh vung vu khi",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All files", "*.*")]
            )
            root.destroy()
            if selected:
                input_path = selected
        except Exception:
            pass

    if not input_path:
        print("[!] Ban chua chon anh nao. Hay chay: python process_image.py \"duong_dan_anh.jpg\"")
        sys.exit(1)

    inp = Path(input_path)

    # 1. Trường hợp là 1 file ảnh
    if inp.is_file():
        print("=" * 70)
        print("📸 CÔNG CỤ XỬ LÝ & KHOANH VÙNG VŨ KHÍ TRÊN ẢNH")
        print(f"File ảnh   : {input_path}")
        print(f"Mô hình AI : {args.weights}")
        print("=" * 70)

        out_path, detections, infer_time = process_single_image(
            image_path=str(inp),
            output_dir=args.output,
            weights_path=args.weights,
            imgsz=args.imgsz,
            conf_thresh=args.conf,
            augment=args.tta,
            weapons_only=args.weapons_only
        )

        print(f"\n[+] Đã xử lý xong trong {infer_time:.1f} ms!")
        print(f"[+] Tìm thấy: {len(detections)} hung khí:")
        for idx, d in enumerate(detections, 1):
            vn = d.get('class_vi') or POLICY.vi_name(d['class'])
            tier = "VŨ KHÍ " if d.get('tier') == TIER_WEAPON else "nghi vấn"
            print(f"   {idx}. [{tier}] {vn} - Độ tin cậy: {d['confidence']*100:.1f}% "
                  f"(ngưỡng lớp {d.get('threshold_used', 0):.2f}) | Độ nét Laplacian: {d['laplacian_var']}")

        print(f"\n[SUCCESS] Ảnh kết quả đã được lưu tại:\n👉 {os.path.abspath(out_path)}")
        print("=" * 70)

        # Mở trực tiếp ảnh đã xử lý lên màn hình để người dùng xem ngay
        try:
            os.startfile(os.path.abspath(out_path))
        except Exception:
            pass

    # 2. Trường hợp là cả 1 thư mục chứa nhiều ảnh
    elif inp.is_dir():
        image_files = [p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        print("=" * 70)
        print(f"📸 CÔNG CỤ QUÉT HÀNG LOẠT {len(image_files)} ẢNH TRONG THƯ MỤC: {input_path}")
        print("=" * 70)

        # Nạp lõi phát hiện MỘT lần cho cả thư mục, thay vì mỗi ảnh một lần.
        detector = WeaponDetector(
            weights=args.weights, imgsz=args.imgsz,
            conf_override=args.conf, weapons_only=args.weapons_only,
            augment=args.tta,
        )
        print(detector.describe())
        total_threats = 0
        for idx, img_p in enumerate(image_files, 1):
            out_p, dets, infer_t = process_single_image(
                image_path=str(img_p),
                output_dir=args.output,
                detector=detector,
                imgsz=args.imgsz,
                conf_thresh=args.conf,
                augment=args.tta,
                weapons_only=args.weapons_only
            )
            total_threats += len(dets)
            print(f" [{idx}/{len(image_files)}] {img_p.name} -> {len(dets)} hung khí ({infer_t:.1f}ms)")

        print(f"\n[SUCCESS] Đã quét xong {len(image_files)} ảnh. Phát hiện tổng cộng: {total_threats} hung khí!")
        print(f"[+] Toàn bộ ảnh đã được lưu tại: {os.path.abspath(args.output)}")
        try:
            os.system(f'explorer.exe /select,"{os.path.abspath(args.output)}"')
        except Exception:
            pass


if __name__ == "__main__":
    main()
