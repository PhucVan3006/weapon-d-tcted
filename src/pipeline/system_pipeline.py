"""
Surveillance System Pipeline Orchestrator.
Spawns and coordinates the 3 asynchronous threads:
  - Thread 1: BufferlessFrameGrabber (Producer)
  - Thread 2: AIInferenceTrackingEngine (Inference + ByteTrack + BestShot)
  - Thread 3: AlertStorageWorker (Consumer - SSD I/O + Database + Notifications)
"""

import time
import queue
import cv2
import yaml
from pathlib import Path

from src.pipeline.frame_grabber import BufferlessFrameGrabber
from src.pipeline.inference_engine import AIInferenceTrackingEngine
from src.pipeline.alert_worker import AlertStorageWorker


class SurveillancePipeline:
    def __init__(self, config_path="configs/system_config.yaml", source_override=None, weights_override=None):
        self.config = self._load_config(config_path)

        q1_size = self.config.get("pipeline", {}).get("frame_queue_size", 2)
        q2_size = self.config.get("pipeline", {}).get("event_queue_size", 50)
        self.frame_queue = queue.Queue(maxsize=q1_size)
        self.alert_queue = queue.Queue(maxsize=q2_size)

        input_source = source_override if source_override is not None else self.config.get("pipeline", {}).get("input_source", 0)
        if isinstance(input_source, str) and input_source.isdigit():
            input_source = int(input_source)

        weights_path = weights_override or self.config.get("inference", {}).get("model_path", "yolo26s.pt")
        imgsz = self.config.get("inference", {}).get("imgsz", 960)
        conf_thresh = self.config.get("inference", {}).get("conf_threshold", 0.30)
        device = self.config.get("inference", {}).get("device", "0")
        half = self.config.get("inference", {}).get("half", True)
        tracking_cfg = self.config.get("tracking", {}) or {}
        alert_cfg = self.config.get("alert_system", {}) or {}
        storage_dir = self.config.get("pipeline", {}).get("save_dir", "storage/alerts")
        db_path = self.config.get("pipeline", {}).get("db_path", "storage/logs/security_events.db")

        print("[+] Instantiating Multi-Threaded Pipeline:")
        print(f"  Thread 1 (Grabber)   : Input source = {input_source}")
        print(f"  Thread 2 (Inference) : Weights = {weights_path}, Imgsz = {imgsz}")
        print(f"  Thread 3 (Worker)    : Alert Storage = {storage_dir}, DB = {db_path}")

        self.thread1 = BufferlessFrameGrabber(source=input_source, frame_queue=self.frame_queue, maxsize=q1_size)
        self.thread2 = AIInferenceTrackingEngine(
            frame_queue=self.frame_queue,
            alert_queue=self.alert_queue,
            weights_path=weights_path,
            imgsz=imgsz,
            conf_threshold=conf_thresh,
            device=device,
            half=half,
            tracking_cfg=tracking_cfg,
            alert_cfg=alert_cfg
        )
        self.thread3 = AlertStorageWorker(
            alert_queue=self.alert_queue,
            storage_dir=storage_dir,
            db_path=db_path
        )

    def _load_config(self, path):
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def start(self, show_display=True):
        print("\n" + "=" * 60)
        print("🚀 STARTING REAL-TIME WEAPON SURVEILLANCE PIPELINE")
        print("⌨️  Phím 'f': Bật / Tắt Toàn Màn Hình (Fullscreen)")
        print("⌨️  Phím 'q' hoặc ESC: Thoát hệ thống")
        print("=" * 60 + "\n")

        self.thread3.start()
        self.thread2.start()
        # Chờ mô hình AI tải xong vào GPU trước khi luồng 1 bắt đầu nạp khung hình
        if hasattr(self.thread2, "ready_event"):
            self.thread2.ready_event.wait(timeout=30.0)
        self.thread1.start()

        win_name = "Production Weapon Surveillance System (YOLO26s + ByteTrack)"
        is_fullscreen = False

        if show_display:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            # Tự động mở cửa sổ to bằng màn hình Full HD
            cv2.resizeWindow(win_name, 1920, 1080)

        try:
            while self.thread2.is_alive() and (self.thread1.is_alive() or not self.frame_queue.empty()):
                annotated = self.thread2.latest_annotated_frame
                if show_display and annotated is not None:
                    cv2.imshow(win_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n[!] Exit requested by user.")
                        break
                    elif key == ord('f') or key == ord('F'):
                        is_fullscreen = not is_fullscreen
                        prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
                else:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[!] Pipeline interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        print("\n[+] Stopping threads safely...")
        self.thread1.stop()
        self.thread2.stop()
        self.thread3.stop()

        cv2.destroyAllWindows()
        print("[SUCCESS] All pipeline workers closed cleanly.")
