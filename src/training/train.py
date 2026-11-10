"""
train.py - Huấn luyện / tinh chỉnh mô hình phát hiện hung khí.

Mọi tham số đọc từ configs/system_config.yaml, mục `training`, và có thể
ghi đè bằng tham số dòng lệnh. Không còn số nào gán cứng rải rác.

BA CHẾ ĐỘ
---------
  --mode full      Huấn luyện từ trọng số COCO (yolo26s.pt). Dùng khi đổi
                   dataset hoặc đổi số lớp. Tốn 15-20 giờ.

  --mode finetune  Tinh chỉnh từ weights/best.pt với learning rate thấp.
                   Dùng sau khi thêm hard negative hoặc thêm ảnh miền mới.
                   Chỉ tốn 1-2 giờ và KHÔNG làm quên kiến thức cũ.
                   >>> Đây là chế độ bạn cần sau khi thu thập hard negative.

  --mode resume    Chạy tiếp lần train bị ngắt giữa chừng.

VÌ SAO THAM SỐ ĐƯỢC ĐẶT NHƯ VẬY
-------------------------------
Hai yêu cầu đặt lên hàng đầu: KHÔNG BỎ SÓT và KHOANH VÙNG CHÍNH XÁC.
Các tham số dưới đây nhắm thẳng vào đó, xếp theo mức tác động:

  imgsz=960        Đòn mạnh nhất cho vật nhỏ ở xa. Một con dao 25 px trong
                   khung 1080p chỉ còn ~15 px sau khi co về 640, gần như
                   mất hết đặc trưng để hồi quy hộp. Ở 960 nó còn ~22 px.
                   Đổi lại: chậm hơn ~2,2 lần và tốn VRAM hơn -> batch giảm.
  box=9.0          Trọng số loss hồi quy hộp (mặc định 7.5). Đây là núm
                   xoay TRỰC TIẾP cho độ sát của hộp bao, kéo mAP50-95 lên.
  mixup=0.0        TẮT. Mixup trộn hai ảnh trong suốt và giữ hộp của CẢ HAI,
                   buộc mô hình đoán hộp cho vật bị chồng mờ -> biên hộp học
                   được nhoè đi. Lợi cho phân lớp, HẠI cho độ sát của hộp.
  copy_paste=0.1   Giảm từ 0.3. Dataset không có mask phân vùng nên thao tác
                   này dán theo hộp chữ nhật, tạo mép cứng giả và hộp lệch.
  close_mosaic=20  Tắt mosaic ở 20 epoch cuối để mô hình kết thúc trên ảnh
                   THẬT, không phải ảnh ghép. Hộp ở cuối ổn định hơn.
  scale=0.5        Giữ. Buộc mô hình thấy cùng một vũ khí ở nhiều kích
                   thước -> bớt phụ thuộc vào "vật to = vũ khí".
  mosaic=1.0       Giữ. Ghép 4 ảnh, vật thể tự nhiên nhỏ lại.
  cls=0.7          Tăng trọng số loss phân lớp (mặc định 0.5). Hệ thống
                   đang nhầm lớp này sang lớp khác, cần phạt nặng hơn.
  cos_lr=True      Giảm learning rate theo cosine -> hội tụ êm ở cuối.

  degrees / shear / perspective = 0
                   CỐ Ý để 0, không phải bỏ quên. Quay ảnh làm hộp bao song
                   song trục PHÌNH TO ra: một con dao mảnh quay 30 độ có hộp
                   rộng gần gấp đôi vật thật. Nhãn thành sai, và hộp lỏng
                   đúng là cái đang cần chữa.

CÁCH DÙNG
---------
    python src/training/train.py --mode finetune
    python src/training/train.py --mode full --epochs 80
    python src/training/train.py --mode resume
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
RUN_DIR = PROJECT_ROOT / "runs" / "detect"
BEST_OUT = PROJECT_ROOT / "weights" / "best.pt"


def load_cfg():
    if not CONFIG.exists():
        print(f"[!] Không tìm thấy {CONFIG}")
        return {}
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_last_checkpoint(name):
    for p in (RUN_DIR / name / "weights" / "last.pt",
              RUN_DIR / "runs" / "detect" / name / "weights" / "last.pt"):
        if p.exists():
            return p
    return None


def build_params(cfg, args):
    t = dict(cfg.get("training") or {})
    data_yaml = args.data or (cfg.get("dataset") or {}).get(
        "yaml_path", "data/weapon_12cls.yaml")
    device = args.device or t.get("device") or ("0" if torch.cuda.is_available() else "cpu")

    p = {
        "data": str(PROJECT_ROOT / data_yaml) if not os.path.isabs(data_yaml) else data_yaml,
        "epochs": args.epochs or t.get("epochs", 80),
        "imgsz": args.imgsz or t.get("imgsz", 960),
        "batch": args.batch or t.get("batch_size", 4),
        "device": device,
        "workers": t.get("workers", 4),
        "optimizer": t.get("optimizer", "AdamW"),
        "lr0": t.get("lr0", 0.0015),
        "lrf": t.get("lrf", 0.01),
        "cos_lr": bool(t.get("cos_lr", True)),
        "warmup_epochs": t.get("warmup_epochs", 3),
        "patience": t.get("patience", 25),
        "save_period": t.get("save_period", 10),
        "close_mosaic": t.get("close_mosaic", 20),

        # Trọng số loss.
        # box cao hơn mặc định (7.5) vì ưu tiên KHOANH VÙNG CHÍNH XÁC.
        # cls cao hơn mặc định (0.5) vì đang nhầm lớp này sang lớp khác.
        "box": t.get("box", 9.0),
        "cls": t.get("cls", 0.7),
        "dfl": t.get("dfl", 1.5),

        # Tăng cường dữ liệu. mixup TẮT và các phép quay/xô lệch để 0 -
        # xem phần giải thích ở đầu file, cả hai đều làm hộp bao lỏng ra.
        "mosaic": t.get("mosaic", 1.0),
        "mixup": t.get("mixup", 0.0),
        "copy_paste": t.get("copy_paste", 0.1),
        "scale": t.get("scale", 0.5),
        "fliplr": t.get("fliplr", 0.5),
        "degrees": t.get("degrees", 0.0),
        "shear": t.get("shear", 0.0),
        "perspective": t.get("perspective", 0.0),
        "translate": t.get("translate", 0.1),
        "hsv_h": t.get("hsv_h", 0.015),
        "hsv_s": t.get("hsv_s", 0.6),
        "hsv_v": t.get("hsv_v", 0.4),

        "project": str(RUN_DIR),
        "name": args.name,
        "exist_ok": True,
        "amp": True,
        "plots": True,
        "val": True,
    }

    if args.mode == "finetune":
        # Learning rate THẤP: học cái mới mà không quên cái cũ.
        # Mosaic/mixup giảm mạnh vì ảnh mới (hard negative, frame thật) cần
        # được học ở dạng nguyên bản, không bị ghép méo.
        p["lr0"] = args.lr0 or 0.0005
        p["epochs"] = args.epochs or 15
        p["warmup_epochs"] = 1
        p["mosaic"] = 0.3
        p["mixup"] = 0.0
        p["copy_paste"] = 0.1
        p["close_mosaic"] = max(3, p["epochs"] // 3)
        p["patience"] = 8
    elif args.lr0:
        p["lr0"] = args.lr0

    return p


def pick_weights(args, cfg):
    if args.weights:
        return args.weights
    if args.mode == "finetune":
        for c in (BEST_OUT,
                  RUN_DIR / "weapon_full_8cls" / "weights" / "best.pt",
                  RUN_DIR / "weapon_full_12cls" / "weights" / "best.pt",
                  RUN_DIR / "runs" / "detect" / "weapon_full_8cls" / "weights" / "best.pt",
                  RUN_DIR / "runs" / "detect" / "weapon_full_12cls" / "weights" / "best.pt"):
            if Path(c).exists():
                return str(c)
        print("[!] Không tìm thấy trọng số đã train để tinh chỉnh.")
        print("    Chuyển sang huấn luyện từ đầu bằng yolo26s.pt")
    return (cfg.get("training") or {}).get("model_weights", "yolo26s.pt")


def main():
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình phát hiện hung khí")
    ap.add_argument("--mode", choices=["full", "finetune", "resume"], default="finetune",
                    help="finetune (mặc định) = tinh chỉnh từ best.pt, nhanh và an toàn")
    ap.add_argument("--weights", "-w", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.name is None:
        args.name = {"full": "weapon_full", "finetune": "weapon_finetune",
                     "resume": "weapon_full"}[args.mode]

    cfg = load_cfg()

    print("=" * 74)
    print(f"   HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN HUNG KHÍ — chế độ: {args.mode.upper()}")
    print("=" * 74)
    if torch.cuda.is_available():
        print(f"   GPU : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    else:
        print("   [!] KHÔNG có GPU — huấn luyện trên CPU sẽ rất lâu.")

    # ---- resume ----
    if args.mode == "resume":
        last = find_last_checkpoint(args.name)
        if not last:
            print(f"[!] Không tìm thấy last.pt của lần train '{args.name}'.")
            return
        print(f"   Chạy tiếp từ: {last}")
        YOLO(str(last)).train(resume=True)
        return

    weights = pick_weights(args, cfg)
    params = build_params(cfg, args)

    if not os.path.exists(params["data"]):
        print(f"[!] Không tìm thấy file dataset: {params['data']}")
        return

    print(f"   Trọng số nền : {weights}")
    print(f"   Dataset      : {params['data']}")
    for k in ("epochs", "imgsz", "batch", "optimizer", "lr0", "cos_lr",
              "box", "cls", "mosaic", "mixup", "copy_paste", "scale",
              "close_mosaic", "patience"):
        print(f"   {k:<13}: {params[k]}")
    print("=" * 74)

    if args.mode == "finetune":
        print("   Chế độ tinh chỉnh: learning rate thấp, mosaic giảm.")
        print("   Mục tiêu là học thêm dữ liệu mới mà KHÔNG quên kiến thức cũ.")
        print("=" * 74)

    model = YOLO(weights)
    try:
        model.train(**params)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        fb = max(2, params["batch"] // 2)
        print(f"\n[!] Tràn VRAM. Thử lại với batch={fb}...")
        params["batch"] = fb
        params["name"] = args.name + "_fallback"
        model.train(**params)

    # Sao chép trọng số tốt nhất ra weights/best.pt
    for cand in (RUN_DIR / params["name"] / "weights" / "best.pt",
                 RUN_DIR / "runs" / "detect" / params["name"] / "weights" / "best.pt"):
        if cand.exists():
            BEST_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, BEST_OUT)
            print(f"\n[XONG] Đã chép trọng số tốt nhất sang: {BEST_OUT}")
            break

    print("\n   Bước tiếp theo:")
    print("     python src/training/evaluate.py --split test")
    print("     python app.py")


if __name__ == "__main__":
    main()
