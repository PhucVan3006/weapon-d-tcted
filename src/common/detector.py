"""
detector.py - LÕI PHÁT HIỆN DUY NHẤT của toàn hệ thống.

VÌ SAO CÓ FILE NÀY
------------------
Trước đây bốn đường chạy (process_image, process_video, media_scanner,
inference_engine) mỗi file tự nạp model, tự tra tên lớp, tự lọc ngưỡng,
tự vẽ hộp. Hệ quả thấy rõ:

  - Mô hình đổi 3 -> 12 lớp: có file được sửa, có file không. Lớp id >= 3
    hiện ra là "weapon" suốt một thời gian dài.
  - Sửa ngưỡng ở một file không ảnh hưởng ba file kia.
  - Bộ lọc hình học thêm vào chỗ này lại thiếu ở chỗ khác.

Nay MỌI quyết định "có phải vũ khí không, tên gì, vẽ thế nào" nằm ở đây.
Các đường chạy chỉ còn lo việc đọc ảnh/video và ghi kết quả.

BỐN TẦNG LỌC, theo đúng thứ tự
------------------------------
  T1. Ngưỡng dò thấp (0.15) ở model.predict()
      -> để ByteTrack giai đoạn 2 có detection mờ mà bám vết
  T2. Ngưỡng RIÊNG từng lớp (súng 0.30 ... xích 0.60)
      -> chi phí hai loại lỗi khác nhau giữa các lớp
  T3. Lọc HÌNH HỌC (hộp không quá 12% khung)
      -> cắt nhiễu bám vào cánh tay, balo, mép tường
  T4. Phân mức vũ khí / nghi vấn
      -> chỉ vũ khí thật mới báo động đỏ
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.common.class_policy import (load_policy, load_inference_cfg,
                                     load_tracking_cfg, TIER_WEAPON)


def safe_crop(frame, box, pad=0):
    """
    Cắt ảnh theo hộp bao, CÓ KẸP BIÊN.

    VÌ SAO CẦN HÀM NÀY: YOLO có thể trả toạ độ hơi âm hoặc vượt khung. Viết
    thẳng `frame[y1:y2, x1:x2]` với y1 âm thì numpy KHÔNG báo lỗi - nó cắt
    từ phía cuối ảnh và trả về một vùng SAI, hoặc một mảng rỗng. Lỗi im
    lặng kiểu đó rất khó thấy: ảnh gửi cho agent là vùng khác hẳn vật thể.

    Trả về mảng rỗng nếu hộp không giao với khung.
    """
    import numpy as _np
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(w, x1 - pad))
    y1 = max(0, min(h, y1 - pad))
    x2 = max(0, min(w, x2 + pad))
    y2 = max(0, min(h, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return _np.empty((0, 0, frame.shape[2] if frame.ndim == 3 else 1),
                         dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


class Detection:
    """Một phát hiện đã qua đủ bốn tầng lọc."""

    __slots__ = ("box", "conf", "cls_id", "cls_name", "vi_name",
                 "tier", "threshold", "area_pct", "track_id")

    def __init__(self, box, conf, cls_id, cls_name, vi_name,
                 tier, threshold, area_pct, track_id=None):
        self.box = [int(v) for v in box]
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.cls_name = cls_name
        self.vi_name = vi_name
        self.tier = tier
        self.threshold = threshold
        self.area_pct = area_pct
        self.track_id = track_id

    @property
    def is_weapon(self):
        return self.tier == TIER_WEAPON

    def as_dict(self):
        return {
            "class": self.cls_name,
            "class_vi": self.vi_name,
            "tier": self.tier,
            "confidence": round(self.conf, 4),
            "threshold_used": self.threshold,
            "area_pct": round(self.area_pct, 3),
            "bbox": self.box,
            "track_id": self.track_id,
        }


class WeaponDetector:
    """
    Lõi phát hiện. Dùng chung cho ảnh, video, camera và giao diện web.

        det = WeaponDetector()
        dets, rejected = det.detect(frame)
        canvas = det.draw(frame, dets)
    """

    DEFAULT_WEIGHTS = [
        "weights/best.pt",
        "runs/detect/weapon_full_8cls/weights/best.pt",
        "runs/detect/weapon_full_12cls/weights/best.pt",
        "runs/detect/runs/detect/weapon_full_8cls/weights/best.pt",
        "runs/detect/runs/detect/weapon_full_12cls/weights/best.pt",
        "yolo26s.pt",
    ]

    def __init__(self, weights=None, imgsz=None, device=None,
                 conf_override=None, weapons_only=False,
                 use_geometry=True, augment=None, config_path=None,
                 iou=None, max_det=None, half=None, agnostic_nms=None):
        from ultralytics import YOLO
        import torch

        self.policy = load_policy(config_path)

        # Tham số suy luận đọc từ mục `inference` của system_config.yaml.
        # Bản trước khai iou_threshold và half ở đó nhưng KHÔNG nơi nào đọc,
        # nên hệ thống chạy bằng mặc định của ultralytics suốt. Nay nối thật.
        inf = load_inference_cfg(config_path)
        self.tracking = load_tracking_cfg(config_path)

        self.imgsz = int(imgsz if imgsz is not None else inf["imgsz"])
        self.iou = float(iou if iou is not None else inf["iou"])
        self.max_det = int(max_det if max_det is not None else inf["max_det"])
        self.agnostic_nms = bool(agnostic_nms if agnostic_nms is not None
                                 else inf["agnostic_nms"])
        self.augment = bool(augment if augment is not None else inf["augment"])

        self.conf_override = conf_override
        self.weapons_only = bool(weapons_only)
        self.use_geometry = bool(use_geometry)

        if device is None:
            device = inf.get("device") or ("0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # FP16 chỉ có nghĩa trên GPU. Bật half khi chạy CPU sẽ lỗi hoặc chậm hơn.
        want_half = inf["half"] if half is None else bool(half)
        self.half = bool(want_half) and str(self.device) != "cpu" and torch.cuda.is_available()

        self.weights = self._resolve_weights(weights)
        self.model = YOLO(self.weights)

        # Tên lớp LUÔN đọc từ file .pt - đây là nguồn sự thật duy nhất.
        # Gán cứng bảng tên ở code chính là lỗi đã khiến mọi lớp id >= 3
        # hiện ra thành "weapon".
        raw = getattr(self.model, "names", None) or {}
        self.names = dict(raw) if isinstance(raw, dict) else dict(enumerate(raw))
        if not self.names:
            raise RuntimeError(
                f"Không đọc được tên lớp từ {self.weights}. File trọng số có thể hỏng.")

        # Ngưỡng truyền vào predict: thấp hơn mọi ngưỡng riêng
        self.predict_conf = (float(conf_override) if conf_override is not None
                             else self.policy.predict_conf())

    # ------------------------------------------------------------------
    def _resolve_weights(self, weights):
        cands = ([weights] if weights else []) + self.DEFAULT_WEIGHTS
        for c in cands:
            if c and os.path.exists(c):
                if weights and c != weights:
                    print(f"[!] Không thấy '{weights}' -> dùng '{c}'")
                return c
        return weights or "yolo26s.pt"

    def describe(self):
        lines = [
            f"Trọng số   : {self.weights}",
            f"Thiết bị   : {self.device}   |   imgsz: {self.imgsz}"
            f"   |   FP16: {'bật' if self.half else 'tắt'}"
            f"   |   TTA: {'bật' if self.augment else 'tắt'}",
            f"NMS        : iou={self.iou:.2f}   |   max_det={self.max_det}"
            f"   |   {'không phân biệt lớp' if self.agnostic_nms else 'riêng từng lớp'}",
            f"Số lớp     : {len(self.names)} — {', '.join(self.names.values())}",
        ]
        if self.conf_override is not None:
            lines.append(f"Ngưỡng     : ÉP CHUNG {self.conf_override:.2f} "
                         f"(bỏ qua ngưỡng riêng từng lớp)")
        else:
            lines.append(self.policy.summary())
        if self.use_geometry:
            lines.append(self.policy.geometry_summary())
        else:
            lines.append("Lọc hình học: TẮT")
        if self.weapons_only:
            lines.append("Chế độ     : CHỈ VŨ KHÍ (bỏ qua nhóm dụng cụ)")
        return "\n".join(lines)

    def class_name(self, cls_id):
        """Tên lớp theo id, đọc từ file .pt. Dùng chung cho mọi đường chạy."""
        return self.names.get(int(cls_id), f"class_{int(cls_id)}")

    # ------------------------------------------------------------------
    # T1 - một lần predict() duy nhất cho mỗi khung hình.
    # Mọi đường chạy đi qua đây, nên tham số iou / max_det / half / TTA
    # chỉ cần khai một chỗ là có hiệu lực ở tất cả.
    # ------------------------------------------------------------------
    def predict_raw(self, frame):
        """
        Trả về list [x1, y1, x2, y2, conf, cls_id], CHƯA lọc gì ngoài
        ngưỡng sàn predict_conf. Đây là dạng ByteTrack cần.
        """
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.predict_conf,
            iou=self.iou,                  # NMS: cao = giữ nhiều hộp chồng nhau
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self.device, augment=self.augment, verbose=False,
        )
        raw = []
        if not results or results[0].boxes is None:
            return raw
        for b in results[0].boxes:
            xyxy = [float(v) for v in b.xyxy[0].cpu().numpy()]
            raw.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                        float(b.conf[0].cpu().numpy()),
                        int(b.cls[0].cpu().numpy())])
        return raw

    # ------------------------------------------------------------------
    # T2 + T3 + T4 - áp lên danh sách thô. Tách riêng để đường chạy có
    # bám vết dùng được hình học mà KHÔNG bị cắt mất detection mờ.
    # ------------------------------------------------------------------
    def classify(self, raw, frame_w, frame_h, apply_class_threshold=True):
        """
        Áp các tầng lọc lên kết quả của predict_raw().

        apply_class_threshold=False -> BỎ QUA T2 (ngưỡng riêng từng lớp).
        Dùng cho đường chạy có ByteTrack: giai đoạn 2 của ByteTrack cần
        detection mờ (conf thấp) để bám vết, việc xác nhận theo ngưỡng lớp
        làm sau, trên vết đã bám. Cắt T2 ngay tại đây là tự bỏ mất vật thể.

        Trả về (kept: list[Detection], rejected: list[(tên, conf, lý do)]).
        """
        kept, rejected = [], []
        area_div = float(frame_w * frame_h) if (frame_w and frame_h) else 0.0

        for x1, y1, x2, y2, conf, cid in raw:
            xyxy = [x1, y1, x2, y2]
            name = self.class_name(cid)
            tier = self.policy.tier(name)
            thr = (self.conf_override if self.conf_override is not None
                   else self.policy.threshold(name))
            area_pct = ((x2 - x1) * (y2 - y1) / area_div * 100.0) if area_div else 0.0

            # T4 - phân mức
            if self.weapons_only and tier != TIER_WEAPON:
                rejected.append((name, conf, "không phải vũ khí thật"))
                continue

            # T2 - ngưỡng riêng từng lớp
            if apply_class_threshold and conf < thr:
                rejected.append((name, conf, f"dưới ngưỡng lớp {thr:.2f}"))
                continue

            # T3 - hình học.
            # TRUYỀN CẢ conf VÀ TÊN LỚP: giới hạn diện tích nội suy theo độ
            # tin cậy, nên cùng một cái hộp to sẽ bị loại khi conf thấp
            # (cánh tay, balo) và được giữ khi conf cao (dao giơ gần camera).
            if self.use_geometry:
                ok, why = self.policy.keep_geometry(xyxy, frame_w, frame_h,
                                                    conf=conf, cls_name=name)
                if not ok:
                    rejected.append((name, conf, why))
                    continue

            kept.append(Detection(
                box=xyxy, conf=conf, cls_id=cid, cls_name=name,
                vi_name=self.policy.vi_name(name), tier=tier,
                threshold=thr, area_pct=area_pct,
            ))

        return kept, rejected

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Đủ bốn tầng lọc trên một khung hình. Dùng cho ảnh tĩnh và cho
        đường chạy không bám vết.
        Trả về (danh_sách_Detection, danh_sách_bị_loại).
        """
        h, w = frame.shape[:2]
        return self.classify(self.predict_raw(frame), w, h,
                             apply_class_threshold=True)

    def detect_for_tracking(self, frame):
        """
        Dành cho đường chạy có ByteTrack. MỘT lần predict().

        Trả về (dets_cho_tracker, rejected) trong đó dets_cho_tracker là
        list [x1, y1, x2, y2, conf, cls_id] đã qua T1 + T3 + T4 nhưng CHƯA
        qua T2. Lọc hình học vẫn áp dụng, nên hộp cỡ cánh tay ở conf thấp
        không bao giờ vào được tracker; còn vũ khí nhỏ ở conf thấp thì vẫn
        vào được và ByteTrack bám vết bình thường.
        """
        h, w = frame.shape[:2]
        kept, rejected = self.classify(self.predict_raw(frame), w, h,
                                       apply_class_threshold=False)
        dets = [[d.box[0], d.box[1], d.box[2], d.box[3], d.conf, d.cls_id]
                for d in kept]
        return dets, rejected

    def make_tracker(self, window_size=8):
        """
        ByteTrack với tham số đọc từ mục `tracking` của system_config.yaml.

        Có hàm này để không file nào còn gán cứng tham số tracker nữa.
        process_video.py từng gọi WeaponByteTracker(track_high_thresh=0.45,
        track_low_thresh=0.15) mà bỏ new_track_thresh, nên bản sửa
        new_track_thresh chỉ có tác dụng ở pipeline camera.
        """
        from src.tracking.weapon_tracker import WeaponByteTracker
        return WeaponByteTracker(window_size=window_size, **self.tracking)

    # ------------------------------------------------------------------
    def draw(self, frame, detections, show_hud=True, copy=True):
        """Vẽ toàn bộ phát hiện lên khung hình. Trả về ảnh đã vẽ."""
        canvas = frame.copy() if copy else frame
        for i, d in enumerate(detections):
            self._draw_one(canvas, d, i)
        if show_hud:
            self._draw_hud(canvas, detections)
        return canvas

    def _draw_one(self, img, d, index=0):
        x1, y1, x2, y2 = d.box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = self.policy.color(d.cls_name)
        weapon = d.is_weapon

        # Vũ khí vẽ đậm, vật nghi vấn vẽ mảnh
        ring_t = 2 if weapon else 1
        
        # Hình bầu dục ôm sát hình dáng vật thể (tránh hình tròn khổng lồ che lấp vật lân cận)
        ax_w = max(w // 2 + 4, 10)
        ax_h = max(h // 2 + 4, 10)
        cv2.ellipse(img, (cx, cy), (ax_w, ax_h), 0, 0, 360, color, ring_t, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)

        # 4 góc ngắm radar bám sát khung bao
        corner = max(8, int(min(w, h) * 0.25))
        t = 2 if weapon else 1
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * corner, py), color, t)
            cv2.line(img, (px, py), (px, py + dy * corner), color, t)

        label = d.vi_name if weapon else f"[NV] {d.vi_name}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        label = f" {label} | {int(d.conf * 100)}% "

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.45 if img.shape[1] < 1000 else 0.55
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        # Đặt nhãn ngay trên đỉnh hộp bao của vật thể (thay vì bị đẩy lên tít trên cao)
        bx1 = max(4, min(x1, img.shape[1] - tw - 6))
        by1 = y1 - th - 8
        if by1 < 4:
            by1 = min(img.shape[0] - th - 8, y1 + 4)

        cv2.rectangle(img, (bx1, by1), (bx1 + tw, by1 + th + 6), color, -1)
        cv2.rectangle(img, (bx1, by1), (bx1 + tw, by1 + th + 6), (255, 255, 255), 1)
        cv2.putText(img, label, (bx1, by1 + th + 2), font, scale,
                    (255, 255, 255), thick, lineType=cv2.LINE_AA)

    def _draw_hud(self, img, detections):
        h, w = img.shape[:2]
        n_weapon = sum(1 for d in detections if d.is_weapon)
        n_suspect = len(detections) - n_weapon

        band = img.copy()
        cv2.rectangle(band, (0, 0), (w, 46), (20, 20, 20), -1)
        cv2.addWeighted(band, 0.72, img, 0.28, 0, img)

        if n_weapon:
            txt = f"CANH BAO: {n_weapon} VU KHI"
            col = (0, 0, 255)
        elif n_suspect:
            txt = f"Theo doi: {n_suspect} vat nghi van"
            col = (0, 215, 255)
        else:
            txt = "An toan - khong phat hien vu khi"
            col = (0, 220, 0)
        cv2.putText(img, txt, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, col, 2, lineType=cv2.LINE_AA)

    # ------------------------------------------------------------------
    @staticmethod
    def summarize(detections):
        """Tóm tắt dạng văn bản, dùng cho giao diện và báo cáo."""
        if not detections:
            return "Không phát hiện vũ khí hay vật nghi vấn nào."
        weapons = [d for d in detections if d.is_weapon]
        suspects = [d for d in detections if not d.is_weapon]
        parts = []
        if weapons:
            parts.append("VŨ KHÍ:")
            for d in sorted(weapons, key=lambda x: -x.conf):
                parts.append(f"   • {d.vi_name} — {d.conf*100:.0f}% "
                             f"(ngưỡng {d.threshold:.2f}, {d.area_pct:.1f}% khung)")
        if suspects:
            parts.append("Vật nghi vấn:")
            for d in sorted(suspects, key=lambda x: -x.conf):
                parts.append(f"   • {d.vi_name} — {d.conf*100:.0f}%")
        return "\n".join(parts)
