"""
Thread 2: AI Inference & Tracking Engine.
Coordinates:
1. YOLO26s qua lõi dùng chung src/common/detector.py (imgsz, iou, max_det,
   FP16 và lọc hình học đều đọc từ configs/system_config.yaml)
2. Dual-Thresholding: dò ở ngưỡng sàn thấp, xác nhận ở ngưỡng riêng từng lớp
3. ByteTrack (Track_High=0.50, Track_Low=0.15)
4. Sliding Window verification (>= 4 of last 8 frames & peak conf >= 0.55)
5. Progressive Best-Shot Selection (Laplacian variance)
Pushes confirmed alert events to Queue 2 for Luồng 3.
"""

import time
import queue
import cv2
import numpy as np
import threading
import warnings
warnings.filterwarnings("ignore")
from src.tracking.weapon_tracker import WeaponByteTracker
from src.bestshot.laplacian_scorer import ProgressiveBestShotManager
from src.common.class_policy import load_policy, TIER_WEAPON
from src.common.detector import WeaponDetector

# Chính sách lớp dùng chung — xem khối class_policy trong system_config.yaml
POLICY = load_policy()


class AIInferenceTrackingEngine(threading.Thread):
    def __init__(
        self,
        frame_queue: queue.Queue,
        alert_queue: queue.Queue,
        weights_path: str = "weights/best.pt",
        imgsz: int = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        device: str = "0",
        class_names: dict = None,
        half: bool = True,
        tracking_cfg: dict = None,
        alert_cfg: dict = None
    ):
        super().__init__(daemon=True, name="Thread-2-AIInference")
        self.frame_queue = frame_queue
        self.alert_queue = alert_queue
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        # BẢNG TÊN MẶC ĐỊNH {0:"gun", 1:"knife", 2:"bat_stick"} ĐÃ BỊ BỎ -
        # ba tên của mô hình 3 lớp cũ. Tên lớp đọc từ file .pt qua detector.
        self.class_names = class_names or {}

        tcfg = tracking_cfg or {}
        acfg = alert_cfg or {}

        self.track_high_thresh = tcfg.get("track_high_thresh", 0.50)
        self.track_low_thresh = tcfg.get("track_low_thresh", 0.15)

        # Ngưỡng dò thật sự truyền vào predict() nay do WeaponDetector tính
        # (= min(detect_conf_floor, ngưỡng lớp nhỏ nhất)). Giữ lại biến này
        # để báo cáo và để trường hợp người gọi ép một ngưỡng chung.
        #
        # Lịch sử: bản đầu dùng conf_threshold=0.30 trong khi tracker khai
        # track_low_thresh=0.15 -> không detection nào dưới 0.30 lọt tới
        # tracker, nên giai đoạn 2 của ByteTrack (thứ sinh ra để cứu vật thể
        # mờ/nhỏ) không có dữ liệu để chạy.
        #
        # conf_threshold = None nghĩa là "dùng ngưỡng riêng từng lớp".
        # Phải kiểm None trước khi gọi min(), nếu không min(None, 0.15) nổ
        # TypeError ngay lúc khởi tạo luồng.
        _candidates = [self.track_low_thresh, POLICY.predict_conf()]
        if self.conf_threshold is not None:
            _candidates.append(float(self.conf_threshold))
        self.detect_conf = min(_candidates)

        # FP16 chỉ có tác dụng trên GPU; bật trên CPU sẽ lỗi hoặc chậm hơn.
        # BẢN CŨ TÍNH self.half RỒI KHÔNG TRUYỀN VÀO predict() - biến chết.
        # Nay truyền qua WeaponDetector, nên nó có hiệu lực thật.
        self.half = bool(half) and str(device).lower() != "cpu"
        self.detector = None      # nạp trong run(), vì ở luồng riêng

        self.min_hits = acfg.get("min_hits", 4)
        self.min_peak_conf = acfg.get("min_peak_conf", 0.55)
        self.strong_hits = acfg.get("strong_hits", 6)
        window_size = acfg.get("sliding_window_size", 8)

        self.tracker = WeaponByteTracker(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            track_buffer=tcfg.get("track_buffer", 30),
            match_thresh=tcfg.get("match_thresh", 0.80),
            new_track_thresh=tcfg.get("new_track_thresh", 0.35),
            window_size=window_size,
        )
        self.best_shot_mgr = ProgressiveBestShotManager(improvement_threshold=1.10)

        self.is_running = True
        self.ready_event = threading.Event()
        self.latest_annotated_frame = None
        self.fps = 0.0

    def run(self):
        print(f"[+] Thread-2: Initializing YOLO model ({self.weights_path}) on device {self.device}...")
        # Lõi phát hiện dùng chung: một lần predict() mỗi khung hình, và
        # iou / max_det / half / lọc hình học đều có hiệu lực ở đây.
        self.detector = WeaponDetector(
            weights=self.weights_path, imgsz=self.imgsz, device=self.device,
            conf_override=self.conf_threshold, iou=self.iou_threshold,
            half=self.half,
        )
        self.imgsz = self.detector.imgsz
        self.class_names = self.detector.names
        print(self.detector.describe())
        self.ready_event.set()

        t_start = time.perf_counter()
        frames_processed = 0

        while self.is_running:
            try:
                frame_idx, frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frames_processed += 1
            t_infer_start = time.perf_counter()

            # 1+2. Một lần predict() qua lõi dùng chung, trả về dạng
            # [[x1, y1, x2, y2, conf, cls_id], ...] cho ByteTrack.
            # Dò ở ngưỡng thấp để giai đoạn 2 của ByteTrack nhận được
            # detection mờ, nhưng ĐÃ áp lọc hình học - bản cũ ở đây không có
            # tầng lọc hình học nào, nên hộp cỡ cánh tay vào thẳng tracker.
            detections, _geo_rej = self.detector.detect_for_tracking(frame)

            # 3. Update ByteTrack
            active_tracks = self.tracker.update(detections)

            # 4. Process each track with Sliding Window + Progressive Best-Shot
            annotated_frame = frame.copy()

            for track in active_tracks:
                if track.time_since_update > 0:
                    continue

                box = track.get_current_box()
                cls_name = self.detector.class_name(track.cls_id)

                # Chỉ vũ khí thật mới được kích hoạt báo động đỏ. Nhóm
                # "nghi vấn" (Crowbar/Stick/Hammer/Spanner/Chain) là dụng cụ
                # - vẫn vẽ khung nhưng không báo động, không lưu bằng chứng.
                is_weapon = POLICY.tier(cls_name) == TIER_WEAPON

                # Ngưỡng đỉnh lấy theo RIÊNG từng lớp thay vì một số cố định
                peak_needed = max(self.min_peak_conf, POLICY.threshold(cls_name)) \
                    if not is_weapon else POLICY.threshold(cls_name)

                is_red_alert = is_weapon and track.evaluate_red_alert(
                    min_hits=self.min_hits,
                    min_peak_conf=peak_needed,
                    strong_hits=self.strong_hits
                )

                if is_red_alert:
                    is_first = not track.alert_triggered
                    track.alert_triggered = True

                    # Evaluate Best-Shot via Laplacian Variance
                    should_update, best_info = self.best_shot_mgr.evaluate_and_update(
                        track_id=track.id,
                        frame=frame,
                        bbox=box,
                        conf=track.conf,
                        is_first_confirmation=is_first
                    )

                    if should_update:
                        event_payload = {
                            "track_id": track.id,
                            "class_name": cls_name,
                            "conf": track.conf,
                            "laplacian_var": best_info["laplacian_var"],
                            "quality_score": best_info["quality_score"],
                            "is_temp": best_info["is_temp"],
                            "update_count": best_info["update_count"],
                            "frame": best_info["frame"],
                            "crop": best_info["crop"],
                            "bbox": box
                        }
                        try:
                            self.alert_queue.put_nowait(event_payload)
                        except queue.Full:
                            pass

                    # BẢNG TÊN GÁN CỨNG ĐÃ BỊ BỎ. Dòng cũ là:
                    #   {"gun": ..., "knife": ..., "bat_stick": ...}.get(cls_name, ...)
                    # Ba tên đó thuộc mô hình 3 lớp cũ, KHÔNG tồn tại trong mô
                    # hình 8 lớp, nên mọi lớp đều rơi vào nhánh cls_name.upper()
                    # và không lớp nào hiện ra tên tiếng Việt. Đây đúng là loại
                    # lỗi mà class_policy.py được tạo ra để diệt: tên lớp chỉ
                    # có MỘT nguồn, là chính sách dùng chung.
                    color = POLICY.color(cls_name)
                    label = f" {POLICY.vi_name(cls_name)} | {int(track.conf * 100)}% [ID:{track.id}] "
                else:
                    color = POLICY.color(cls_name)
                    label = f" [NV] {POLICY.vi_name(cls_name)} | {int(track.conf * 100)}% "

                x1, y1, x2, y2 = [int(v) for v in box]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2 + 8

                # 1. Khoanh tròn mục tiêu
                cv2.circle(annotated_frame, (cx, cy), radius, color, 2, lineType=cv2.LINE_AA)
                cv2.circle(annotated_frame, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)

                # 2. Khung viền và huy hiệu tên
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                by1 = max(0, cy - radius - th - 10)
                cv2.rectangle(annotated_frame, (cx - tw // 2, by1), (cx + tw // 2, by1 + th + 6), color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (cx - tw // 2, by1 + th + 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA
                )

            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self.fps = frames_processed / elapsed
                frames_processed = 0
                t_start = time.perf_counter()

            cv2.putText(
                annotated_frame,
                f"FPS: {self.fps:.1f} | Active Tracks: {len(active_tracks)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            self.latest_annotated_frame = annotated_frame
            self.frame_queue.task_done()

    def stop(self):
        self.is_running = False
