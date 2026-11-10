"""
Evaluation & Hard Negative Mining Module (Stage 4).

- Đánh giá trọng số trên tập test/valid: mAP@50, mAP@50:95, Precision, Recall
  CHI TIẾT TỚI TỪNG LỚP trong 12 lớp - đây là bảng cần đưa vào báo cáo.
- Dò ngưỡng tối ưu riêng cho từng lớp theo F1, xuất ra khối YAML dán thẳng
  được vào configs/system_config.yaml.
- Khai thác false positive từ video giám sát thật để làm hard negative.

NÂNG CẤP v2
-----------
1. In bảng P/R/mAP cho TỪNG LỚP. Bản cũ chỉ in 4 số trung bình, không
   cho biết lớp nào yếu - mà chính lớp yếu mới là chỗ cần sửa.
2. Thêm --tta (augment=True): suy luận đa tỉ lệ + lật ảnh, thường thêm
   1-2 điểm mAP. Chậm hơn ~3 lần nhưng chỉ chạy lúc đánh giá.
3. Thêm --tune-thresholds: quét ngưỡng 0.05..0.90 cho từng lớp, chọn
   điểm F1 cao nhất, in ra khối YAML sẵn sàng dán vào config.
4. Sửa mine_false_positives: bản cũ mặc định trỏ tới
   runs/detect/weapon_yolo26s/weights/best.pt - đường dẫn của lần train
   cũ đã không còn tồn tại.

CÁCH DÙNG
---------
    python src/training/evaluate.py                       # đánh giá tập test
    python src/training/evaluate.py --tta                 # bật TTA, số liệu cao nhất
    python src/training/evaluate.py --tune-thresholds     # dò ngưỡng tối ưu từng lớp
    python src/training/evaluate.py --mine-video clip.mp4 # khai thác hard negative
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _resolve_weights(weights: str) -> str:
    """Tìm trọng số theo thứ tự ưu tiên, báo rõ nếu phải lùi phương án."""
    candidates = [
        weights,
        "weights/best.pt",
        "runs/detect/runs/detect/weapon_full_12cls/weights/best.pt",
        "yolo26s.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            if c != weights:
                print(f"[!] Không thấy '{weights}' -> dùng '{c}'")
            return c
    return weights


def _get_default_data_yaml():
    cfg_file = PROJECT_ROOT / "configs" / "system_config.yaml"
    if cfg_file.exists():
        try:
            import yaml
            with open(cfg_file, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
                yp = (c.get("dataset") or {}).get("yaml_path")
                if yp and (PROJECT_ROOT / yp).exists():
                    return yp
        except Exception:
            pass
    if (PROJECT_ROOT / "data" / "weapon_8cls.yaml").exists():
        return "data/weapon_8cls.yaml"
    return "data/weapon_12cls.yaml"


def evaluate_model(
    weights: str = "weights/best.pt",
    data_yaml: str = None,
    imgsz: int = 960,
    split: str = "test",
    device: str = "0",
    augment: bool = False
):
    if data_yaml is None:
        data_yaml = _get_default_data_yaml()
    weights = _resolve_weights(weights)

    print("=" * 78)
    print("   ĐÁNH GIÁ MÔ HÌNH & PHÂN TÍCH MA TRẬN NHẦM LẪN")
    print(f"   Trọng số : {weights}")
    print(f"   Tập      : {split}   |   imgsz: {imgsz}   |   TTA: {'BẬT' if augment else 'tắt'}")
    print("=" * 78)

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        split=split,
        device=device,
        augment=augment,
        verbose=True
    )

    box = metrics.box
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    print("\n" + "=" * 78)
    print("   KẾT QUẢ THEO TỪNG LỚP")
    print("=" * 78)
    print(f"{'Lớp':<14}{'Precision':>11}{'Recall':>10}{'mAP@0.5':>11}{'mAP@0.5:0.95':>14}")
    print("-" * 78)

    rows = []
    try:
        # box.ap_class_index cho biết thứ tự lớp trong các mảng p/r/ap50/ap
        for i, cls_idx in enumerate(box.ap_class_index):
            name = names.get(int(cls_idx), f"class_{cls_idx}")
            p, r = float(box.p[i]), float(box.r[i])
            ap50, ap = float(box.ap50[i]), float(box.ap[i])
            rows.append((name, p, r, ap50, ap))
            print(f"{name:<14}{p:>11.4f}{r:>10.4f}{ap50:>11.4f}{ap:>14.4f}")
    except Exception as e:
        print(f"[!] Không đọc được số liệu từng lớp: {e}")

    print("-" * 78)
    print(f"{'TRUNG BÌNH':<14}{box.mp:>11.4f}{box.mr:>10.4f}{box.map50:>11.4f}{box.map:>14.4f}")
    print("=" * 78)

    if rows:
        worst_recall = sorted(rows, key=lambda x: x[2])[:3]
        worst_prec = sorted(rows, key=lambda x: x[1])[:3]
        print("\n   BA LỚP BỎ SÓT NHIỀU NHẤT (recall thấp nhất):")
        for n, p, r, a50, a in worst_recall:
            print(f"     {n:<12} recall {r:.3f}  -> cân nhắc HẠ ngưỡng lớp này")
        print("\n   BA LỚP BÁO NHẦM NHIỀU NHẤT (precision thấp nhất):")
        for n, p, r, a50, a in worst_prec:
            print(f"     {n:<12} precision {p:.3f}  -> cân nhắc NÂNG ngưỡng lớp này")
        print()

    return metrics


def tune_thresholds(
    weights: str = "weights/best.pt",
    data_yaml: str = None,
    imgsz: int = 960,
    split: str = "val",
    device: str = "0",
    beta: float = 1.0
):
    if data_yaml is None:
        data_yaml = _get_default_data_yaml()
    """
    Dò ngưỡng tối ưu cho TỪNG LỚP bằng F-beta.

    beta > 1 ưu tiên recall (ít bỏ sót hơn), beta < 1 ưu tiên precision.
    Với hệ thống an ninh, beta = 2 là lựa chọn hợp lý cho nhóm vũ khí:
    bỏ sót một khẩu súng tốn kém hơn nhiều so với một lần báo nhầm.
    """
    weights = _resolve_weights(weights)
    print("=" * 78)
    print(f"   DÒ NGƯỠNG TỐI ƯU TỪNG LỚP (F-beta, beta={beta})")
    print(f"   beta>1 ưu tiên GIẢM BỎ SÓT, beta<1 ưu tiên GIẢM BÁO NHẦM")
    print("=" * 78)

    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    metrics = model.val(data=data_yaml, imgsz=imgsz, split=split,
                        device=device, verbose=False)
    box = metrics.box

    # box.p_curve / r_curve có dạng (n_class, 1000) theo trục conf 0..1
    try:
        import numpy as np
        px = getattr(box, "px", None)
        if px is None:
            px = np.linspace(0, 1, box.p_curve.shape[1])
        b2 = beta ** 2
        print(f"\n{'Lớp':<14}{'ngưỡng tốt nhất':>18}{'P':>9}{'R':>9}{'F-beta':>10}")
        print("-" * 78)
        best_map = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            name = names.get(int(cls_idx), f"class_{cls_idx}")
            p_c = box.p_curve[i]
            r_c = box.r_curve[i]
            f = (1 + b2) * p_c * r_c / (b2 * p_c + r_c + 1e-16)
            j = int(f.argmax())
            best_map[name] = round(float(px[j]), 2)
            print(f"{name:<14}{px[j]:>18.2f}{p_c[j]:>9.3f}{r_c[j]:>9.3f}{f[j]:>10.3f}")
        print("-" * 78)

        print("\n   DÁN KHỐI SAU VÀO configs/system_config.yaml, mục class_policy.thresholds:\n")
        print("  thresholds:")
        print("    default: 0.40")
        for n, v in best_map.items():
            print(f"    {n}: {v:.2f}")
        print()
        print("   LƯU Ý: đây là ngưỡng tối ưu theo F-beta trên tập val. Hãy đối chiếu")
        print("   với mức nguy hiểm trước khi áp dụng - súng/lựu đạn nên giữ ngưỡng")
        print("   thấp hơn giá trị F tối ưu, còn dụng cụ thì nên cao hơn.")
        return best_map
    except Exception as e:
        print(f"[!] Không dò được ngưỡng: {e}")
        return {}


def mine_false_positives_from_video(
    video_path: str,
    weights: str = "weights/best.pt",
    output_dir: str = "data/hard_negatives/images",
    output_lbl_dir: str = "data/hard_negatives/labels",
    conf_threshold: float = 0.40,
    frame_stride: int = 5,
    save_prefix: str = "mined_fp_"
):
    """
    Lưu các frame mà mô hình CÓ phát hiện, để làm hard negative.

    CẢNH BÁO: script ghi nhãn RỖNG cho mọi frame lưu được, tức là dạy mô
    hình rằng "ở đây không có gì". Nếu video thật sự có hung khí thì bạn
    đang dạy mô hình BỎ QUA hung khí. Vì vậy mặc định ghi ra thư mục
    riêng data/hard_negatives/ để BẠN XEM LẠI THỦ CÔNG trước khi trộn vào
    tập train - bản cũ ghi thẳng vào train/images, rất nguy hiểm.
    """
    weights = _resolve_weights(weights)
    print(f"[+] Khai thác false positive từ: {video_path}")
    print(f"[!] Nhớ XEM LẠI THỦ CÔNG trước khi trộn vào tập train.")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_fp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            saved_fp += 1
            stem = f"{save_prefix}{saved_fp:05d}"
            cv2.imwrite(os.path.join(output_dir, f"{stem}.jpg"), frame)
            open(os.path.join(output_lbl_dir, f"{stem}.txt"), "w", encoding="utf-8").close()

    cap.release()
    print(f"[XONG] Trích {saved_fp} frame vào {output_dir}")
    print(f"       Xem lại, XOÁ frame nào thật sự có hung khí, rồi mới trộn vào train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình và khai thác hard negative")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--data", type=str, default=None,
                        help="File cấu hình dataset (mặc định đọc từ system_config.yaml hoặc data/weapon_8cls.yaml)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Phải KHỚP imgsz lúc train, nếu không số đo sẽ lệch")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tta", action="store_true",
                        help="Bật Test-Time Augmentation: +1-2 mAP, chậm hơn ~3 lần")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Dò ngưỡng tối ưu từng lớp theo F-beta")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="beta của F-beta khi dò ngưỡng (mặc định 2.0 = ưu tiên giảm bỏ sót)")
    parser.add_argument("--mine-video", type=str, default=None,
                        help="Đường dẫn video để khai thác false positive")
    args = parser.parse_args()

    if args.mine_video:
        mine_false_positives_from_video(args.mine_video, weights=args.weights)
    elif args.tune_thresholds:
        tune_thresholds(weights=args.weights, data_yaml=args.data, imgsz=args.imgsz,
                        split="val", device=args.device, beta=args.beta)
    else:
        evaluate_model(weights=args.weights, data_yaml=args.data, imgsz=args.imgsz,
                       split=args.split, device=args.device, augment=args.tta)
