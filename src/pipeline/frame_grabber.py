"""
Thread 1: Bufferless Frame Grabber (Producer).
Continuously grabs frames from Camera/RTSP/Video without buffering delay.
Uses queue.Queue(maxsize=2). Drops oldest frame when queue is full.
"""

import os
import time
import queue
import threading
import cv2


class BufferlessFrameGrabber(threading.Thread):
    def __init__(self, source=0, frame_queue=None, maxsize=2, loop=True):
        super().__init__(daemon=True, name="Thread-1-FrameGrabber")
        self.source = source
        self.frame_queue = frame_queue if frame_queue is not None else queue.Queue(maxsize=maxsize)
        self.cap = None
        self.is_running = False
        self.fps = 0.0
        self.total_frames = 0
        self.dropped_frames = 0
        self.loop = loop
        self.is_file = isinstance(self.source, str) and os.path.exists(self.source)

    def run(self):
        is_cam = not self.is_file and (isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()))
        if is_cam:
            cam_idx = int(self.source)
            # Dùng cv2.CAP_DSHOW trên Windows để mở khóa góc rộng 16:9 tối đa của cảm biến
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)
            # Thiết lập chuẩn 16:9 Widescreen (1280x720) để thu trọn vẹn góc rộng, không bị cắt xén (crop) 4:3
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[!] Thread-1: Could not open video source: {self.source}")
            self.is_running = False
            return

        # Tính toán khoảng cách giữa các khung hình nếu là video file để phát đúng tốc độ thật
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0 or video_fps > 120:
            video_fps = 25.0
        frame_interval = 1.0 / video_fps

        self.is_running = True
        t_start = time.perf_counter()
        frame_count = 0

        while self.is_running and self.cap.isOpened():
            t_frame_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file and self.loop:
                    # Lặp lại video để tiếp tục giám sát thử nghiệm
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break

            frame_count += 1
            self.total_frames += 1

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((frame_count, frame))
            except queue.Full:
                self.dropped_frames += 1

            # Điều tiết tốc độ nếu là file video để không bị đọc hết trong 0.2 giây
            if self.is_file:
                t_used = time.perf_counter() - t_frame_start
                sleep_time = frame_interval - t_used
                if sleep_time > 0:
                    time.sleep(sleep_time)

            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

        self.is_running = False
        if self.cap:
            self.cap.release()

    def stop(self):
        self.is_running = False
