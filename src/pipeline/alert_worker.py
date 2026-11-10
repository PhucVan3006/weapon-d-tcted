"""
Thread 3: Alert & Storage Worker (Consumer).
Decouples SSD I/O, Database Logging, and Alert Dispatch from AI Inference.
- Saves Best-Shot evidence photos (JPEG high quality).
- Progressively updates evidence photos when sharper images arrive.
- Stores audit log in SQLite database (security_events.db).
- Dispatches notifications without freezing the video stream.
"""

import os
import cv2
import json
import time
import queue
import sqlite3
import threading
from datetime import datetime

try:
    from src.agent.security_agent import SecurityAnalystAgent
except ImportError:
    SecurityAnalystAgent = None


class AlertStorageWorker(threading.Thread):
    def __init__(self, alert_queue, storage_dir="storage/alerts", db_path="storage/logs/security_events.db", enable_agent=True):
        super().__init__(daemon=True, name="Thread-3-AlertWorker")
        self.alert_queue = alert_queue
        self.storage_dir = storage_dir
        self.db_path = db_path
        self.is_running = True
        self.processed_events = 0
        self.enable_agent = bool(enable_agent) and SecurityAnalystAgent is not None
        self.agent = None          # nạp lười, trong luồng của chính mình
        # Mỗi track CHỈ gọi agent MỘT lần.
        # Bản cũ gọi ở mọi lần cập nhật best-shot (up_count 1, 2, 3...), nên
        # một vũ khí duy nhất tốn nhiều lần gọi API trả phí và mỗi lần lại
        # chặn luồng này thêm vài giây.
        self._agent_done = {}
        self.agent_suppressed = 0

        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_agent(self):
        if not self.enable_agent:
            return None
        if self.agent is None:
            self.agent = SecurityAnalystAgent(quiet=True)
        return self.agent if self.agent.is_available() else None

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weapon_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                track_id INTEGER,
                class_name TEXT,
                confidence REAL,
                laplacian_var REAL,
                quality_score REAL,
                is_temporary BOOLEAN,
                update_count INTEGER,
                image_path TEXT,
                crop_path TEXT
            )
        """)
        # Di trú: thêm cột cho kết luận của agent nếu DB cũ chưa có.
        # ALTER TABLE ADD COLUMN trên cột đã tồn tại sẽ lỗi, nên kiểm trước.
        cursor.execute("PRAGMA table_info(weapon_alerts)")
        have = {row[1] for row in cursor.fetchall()}
        for col, decl in (("agent_threat_level", "INTEGER"),
                          ("agent_is_false_alarm", "BOOLEAN"),
                          ("agent_summary", "TEXT"),
                          ("alert_downgraded", "BOOLEAN")):
            if col not in have:
                cursor.execute(f"ALTER TABLE weapon_alerts ADD COLUMN {col} {decl}")
        conn.commit()
        conn.close()

    def run(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        while self.is_running or not self.alert_queue.empty():
            try:
                event = self.alert_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if event is None:
                break

            self._handle_event(event, cursor, conn)
            self.alert_queue.task_done()

        conn.close()

    def _handle_event(self, event, cursor, conn):
        self.processed_events += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        track_id = event["track_id"]
        cls_name = event["class_name"]
        conf = event["conf"]
        lap_var = event["laplacian_var"]
        q_score = event["quality_score"]
        is_temp = event.get("is_temp", False)
        up_count = event.get("update_count", 1)

        img_filename = f"ALERT_ID{track_id}_{cls_name}_{time_tag}.jpg"
        crop_filename = f"CROP_ID{track_id}_{cls_name}_{time_tag}.jpg"

        full_img_path = os.path.join(self.storage_dir, img_filename)
        crop_path = os.path.join(self.storage_dir, crop_filename)

        frame = event.get("frame")
        crop = event.get("crop")
        wrote_full = frame is not None and frame.size > 0
        wrote_crop = crop is not None and crop.size > 0
        if wrote_full:
            cv2.imwrite(full_img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if wrote_crop:
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # --- AI Agent: chạy TRƯỚC khi in cảnh báo, và chỉ MỘT lần mỗi track.
        # Bằng chứng vẫn được lưu ở trên dù agent nói gì - agent chỉ được HẠ
        # MỨC cảnh báo, không được xoá phát hiện. Xem verdict() trong
        # security_agent.py để biết lý do.
        v = None
        agent = self._get_agent()
        if agent and wrote_full and track_id not in self._agent_done:
            try:
                report = agent.analyze_threat(frame, crop=crop if wrote_crop else None,
                                              class_name=cls_name, confidence=conf,
                                              track_id=track_id)
                v = agent.verdict(report)
                self._agent_done[track_id] = v
                json_path = os.path.join(
                    self.storage_dir,
                    f"ALERT_ID{track_id}_{cls_name}_{time_tag}_agent.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"   [!] AI Agent error: {e}")
        elif track_id in self._agent_done:
            v = self._agent_done[track_id]   # dùng lại kết luận, không gọi lại

        downgraded = bool(v and v["downgrade_alert"])
        if downgraded:
            self.agent_suppressed += 1

        cursor.execute("""
            INSERT INTO weapon_alerts 
            (timestamp, track_id, class_name, confidence, laplacian_var, quality_score,
             is_temporary, update_count, image_path, crop_path,
             agent_threat_level, agent_is_false_alarm, agent_summary, alert_downgraded)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_str, track_id, cls_name, conf, lap_var, q_score, is_temp, up_count,
              full_img_path if wrote_full else None,
              crop_path if wrote_crop else None,
              v["threat_level"] if v else None,
              v["is_false_alarm"] if v else None,
              v["summary_vi"] if v else None,
              downgraded))
        conn.commit()

        status_label = "[TEMP EVIDENCE]" if is_temp else f"[PROGRESSIVE BEST-SHOT #{up_count}]"
        if downgraded:
            # Hạ còi: vẫn ghi nhận, vẫn lưu bằng chứng, KHÔNG báo động đỏ.
            print(f"\n🟢 {status_label} HA MUC - AGENT XAC DINH KHONG PHAI MOI DE DOA: "
                  f"{cls_name.upper()} (Track #{track_id}) | Conf: {conf:.2f} | {img_filename}")
        else:
            print(f"\n🚨 {status_label} RED ALERT: {cls_name.upper()} (Track #{track_id}) "
                  f"| Conf: {conf:.2f} | Laplacian: {lap_var:.1f} | Saved: {img_filename}")

        if v:
            print(f"   🤖 AI AGENT [{v['badge']}] {v['summary_vi']}")
            print(f"      Boi canh : {v['context']}")
            print(f"      Hanh vi  : {v['person_behavior']}")
            print(f"      Khuyen nghi: {v['action_recommended']}")
            print(f"      ({v['latency_ms']} ms, {v['model']})")

    def stop(self):
        self.is_running = False
        self.alert_queue.put(None)
