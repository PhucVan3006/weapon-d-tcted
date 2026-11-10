"""
Media Scanner Module - Comprehensive Offline Image & Video Processing Engine.
Provides thorough, frame-by-frame analysis without dropping frames:
1. Image Scanning (Single image or image directories):
   - Weapon detection, Laplacian variance calculation, quality scoring.
   - Bounding box annotation, crop extraction, CSV/JSON report.
2. Video Scanning (Single video or video directories):
   - Frame-by-frame analysis with ByteTrack.
   - Dual-thresholding + Sliding Window (4/8 frames) threat verification.
   - Progressive Best-Shot selection (Laplacian sharpness).
   - Renders and exports annotated video (.mp4).
   - Generates event timeline report with timestamps, peak confidence, and evidence photos.
"""

import os
import cv2
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from src.tracking.weapon_tracker import WeaponByteTracker
from src.bestshot.laplacian_scorer import ProgressiveBestShotManager, calculate_quality_score, compute_laplacian_variance
from src.common.class_policy import load_policy, TIER_WEAPON
from src.common.detector import WeaponDetector

# Chính sách lớp dùng chung — xem khối class_policy trong system_config.yaml
POLICY = load_policy()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


class MediaScanner:
    def __init__(
        self,
        weights_path: str = "weights/best.pt",
        imgsz: int = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        device: str = None,
        class_names: dict = None
    ):
        # BẢNG TÊN MẶC ĐỊNH {0:"gun", 1:"knife", 2:"bat_stick"} ĐÃ BỊ BỎ.
        # Ba tên đó thuộc mô hình 3 lớp cũ. Nếu file .pt vì lý do nào đó
        # không đọc được tên lớp, bản cũ sẽ IM LẶNG dùng ba tên sai thay vì
        # báo lỗi - đúng cơ chế đã khiến mọi lớp id >= 3 hiện ra là "weapon".
        # WeaponDetector báo RuntimeError trong trường hợp đó.
        self.weights_path = weights_path
        self.detector = WeaponDetector(
            weights=weights_path, imgsz=imgsz, device=device,
            conf_override=conf_threshold, iou=iou_threshold,
        )
        self.imgsz = self.detector.imgsz
        self.iou_threshold = self.detector.iou
        self.device = self.detector.device
        self.class_names = self.detector.names
        print("[+] Media Scanner dùng lõi phát hiện dùng chung:")
        print(self.detector.describe())

    def scan_image(self, image_path: str, output_dir: str, save_crops: bool = True) -> dict:
        """
        Scans a single image, annotates detections, computes Laplacian sharpness,
        and saves outputs.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"file": image_path, "status": "error", "message": "Failed to read image"}

        h, w = img.shape[:2]
        base_name = Path(image_path).stem

        # Lõi phát hiện lo đủ bốn tầng lọc. BẢN CŨ Ở ĐÂY KHÔNG CÓ TẦNG LỌC
        # HÌNH HỌC NÀO - nó chỉ lọc ngưỡng lớp - nên hộp bám vào cánh tay và
        # balo đi thẳng vào kết quả quét.
        dets, _rejected = self.detector.detect(img)

        detections = []
        annotated = img.copy()

        for idx, d in enumerate(dets):
            xyxy = [int(v) for v in d.box]
            cls_name = d.cls_name

            # Compute Laplacian variance for this detection
            q_score, lap_var, roi = calculate_quality_score(img, xyxy, d.conf)

            # Save crop
            crop_path = None
            if save_crops and roi.size > 0:
                crops_dir = os.path.join(output_dir, "crops")
                os.makedirs(crops_dir, exist_ok=True)
                crop_filename = f"{base_name}_det{idx}_{cls_name}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)
                cv2.imwrite(crop_path, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Draw annotation - màu và tên tiếng Việt lấy từ chính sách chung
            color = POLICY.color(cls_name)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f"{POLICY.vi_name(cls_name)} {d.conf:.2f} | Lap:{lap_var:.1f}"
            cv2.putText(annotated, label, (xyxy[0], max(20, xyxy[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "detection_id": idx + 1,
                "class_id": d.cls_id,
                "class_name": cls_name,
                "class_vi": d.vi_name,
                "tier": d.tier,
                "confidence": round(d.conf, 4),
                "threshold_used": d.threshold,
                "area_pct": round(d.area_pct, 3),
                "laplacian_variance": round(lap_var, 2),
                "quality_score": round(q_score, 2),
                "bbox": xyxy,
                "crop_path": crop_path
            })

        # Save annotated image
        images_dir = os.path.join(output_dir, "annotated_images")
        os.makedirs(images_dir, exist_ok=True)
        annotated_path = os.path.join(images_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return {
            "file": image_path,
            "status": "success",
            "threat_count": len(detections),
            "detections": detections,
            "annotated_path": annotated_path
        }

    def scan_video(
        self,
        video_path: str,
        output_dir: str,
        save_annotated_video: bool = True,
        save_crops: bool = True
    ) -> dict:
        """
        Scans a video file frame-by-frame with full tracking, sliding-window alert evaluation,
        and progressive best-shot selection.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"file": video_path, "status": "error", "message": "Failed to open video"}

        base_name = Path(video_path).stem
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        video_out_path = None
        writer = None
        if save_annotated_video:
            videos_dir = os.path.join(output_dir, "annotated_videos")
            os.makedirs(videos_dir, exist_ok=True)
            video_out_path = os.path.join(videos_dir, f"{base_name}_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

        # Evidence directories
        alerts_dir = os.path.join(output_dir, "best_shot_evidence", base_name)
        os.makedirs(alerts_dir, exist_ok=True)

        # Tham số tracker đọc từ mục `tracking` của system_config.yaml,
        # thay vì gán cứng và bỏ quên new_track_thresh như bản cũ.
        tracker = self.detector.make_tracker()
        best_shot_mgr = ProgressiveBestShotManager(improvement_threshold=1.10)

        # Track-level event summary: track_id -> event details
        events_summary = {}

        print(f"\n[+] Scanning Video: {video_path} ({total_frames} frames @ {fps:.1f} FPS)...")
        frame_idx = 0
        t0 = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            timestamp_sec = frame_idx / fps
            timestamp_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}"

            # 1+2. Một lần predict() qua lõi dùng chung.
            # TRƯỚC ĐÂY GÁN CỨNG conf=0.30: mọi lớp có ngưỡng riêng bằng
            # 0.30 (Pistol, Long_gun, Grenade) bị chặn ngay từ tầng dò, và
            # ByteTrack giai đoạn 2 không còn detection mờ nào để bám vết.
            # Và bản cũ không có tầng lọc hình học nào ở đây.
            detections, _geo_rej = self.detector.detect_for_tracking(frame)

            # 3. Update ByteTrack
            active_tracks = tracker.update(detections)

            # 4. Evaluate Sliding Window & Best-Shot
            annotated_frame = frame.copy() if save_annotated_video else None

            for track in active_tracks:
                if track.time_since_update > 0:
                    continue

                box = track.get_current_box()
                cls_name = self.detector.class_name(track.cls_id)

                # Xác nhận theo ngưỡng RIÊNG của lớp thay vì 0.55 cố định.
                # Vật ở mức "nghi vấn" (dụng cụ) không kích hoạt báo động đỏ.
                if POLICY.tier(cls_name) != TIER_WEAPON:
                    continue
                is_red = track.evaluate_red_alert(
                    min_hits=4,
                    min_peak_conf=POLICY.threshold(cls_name)
                )

                if is_red:
                    is_first = not track.alert_triggered
                    track.alert_triggered = True

                    should_update, best_info = best_shot_mgr.evaluate_and_update(
                        track_id=track.id,
                        frame=frame,
                        bbox=box,
                        conf=track.conf,
                        is_first_confirmation=is_first
                    )

                    evidence_path = os.path.join(alerts_dir, f"Track_{track.id}_{cls_name}_bestshot.jpg")
                    crop_evidence_path = os.path.join(alerts_dir, f"Track_{track.id}_{cls_name}_crop.jpg")

                    if should_update:
                        # Save / update best-shot photo
                        cv2.imwrite(evidence_path, best_info["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if best_info["crop"].size > 0:
                            cv2.imwrite(crop_evidence_path, best_info["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])

                    # Record event metadata
                    if track.id not in events_summary:
                        events_summary[track.id] = {
                            "track_id": track.id,
                            "class_name": cls_name,
                            "first_seen_frame": frame_idx,
                            "first_seen_time": timestamp_str,
                            "last_seen_frame": frame_idx,
                            "last_seen_time": timestamp_str,
                            "peak_conf": round(track.conf, 4),
                            "best_laplacian_var": round(best_info["laplacian_var"], 2),
                            "quality_score": round(best_info["quality_score"], 2),
                            "updates_count": best_info.get("update_count", 1),
                            "evidence_image": evidence_path,
                            "crop_image": crop_evidence_path
                        }
                    else:
                        ev = events_summary[track.id]
                        ev["last_seen_frame"] = frame_idx
                        ev["last_seen_time"] = timestamp_str
                        if track.conf > ev["peak_conf"]:
                            ev["peak_conf"] = round(track.conf, 4)
                        if best_info["laplacian_var"] > ev["best_laplacian_var"]:
                            ev["best_laplacian_var"] = round(best_info["laplacian_var"], 2)
                            ev["quality_score"] = round(best_info["quality_score"], 2)
                        ev["updates_count"] = best_info.get("update_count", ev["updates_count"])

                    # Draw red alert overlay on frame
                    if annotated_frame is not None:
                        color = (0, 0, 255)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                        label = f"RED ALERT: {cls_name.upper()} #{track.id} ({track.conf:.2f})"
                        cv2.putText(annotated_frame, label, (box[0], max(20, box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    if annotated_frame is not None:
                        color = (0, 215, 255)
                        cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                        label = f"Track #{track.id}: {cls_name} ({track.conf:.2f})"
                        cv2.putText(annotated_frame, label, (box[0], max(20, box[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if writer is not None and annotated_frame is not None:
                # Add status banner
                banner_text = f"Time: {timestamp_str} | Threats: {len(events_summary)}"
                cv2.putText(annotated_frame, banner_text, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                writer.write(annotated_frame)

            # Progress log every 100 frames
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                elapsed = time.perf_counter() - t0
                speed = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_idx}/{total_frames} frames ({pct:.1f}%) - {speed:.1f} FPS | Detected Threats: {len(events_summary)}")

        cap.release()
        if writer is not None:
            writer.release()

        return {
            "file": video_path,
            "status": "success",
            "total_frames": total_frames,
            "fps": fps,
            "confirmed_threats": len(events_summary),
            "events": list(events_summary.values()),
            "annotated_video": video_out_path,
            "evidence_dir": alerts_dir
        }

    def process_path(
        self,
        input_path: str,
        output_dir: str = "storage/scan_results",
        save_annotated_video: bool = True,
        save_crops: bool = True
    ) -> list:
        """
        Processes a file or directory of images/videos.
        Generates combined JSON & CSV summary reports.
        """
        os.makedirs(output_dir, exist_ok=True)
        p = Path(input_path)

        all_results = []

        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                res = self.scan_image(str(p), output_dir, save_crops=save_crops)
                all_results.append(res)
            elif ext in VIDEO_EXTENSIONS:
                res = self.scan_video(str(p), output_dir, save_annotated_video=save_annotated_video, save_crops=save_crops)
                all_results.append(res)
            else:
                print(f"[-] Unsupported file type: {ext}")
        elif p.is_dir():
            files = sorted(list(p.iterdir()))
            for f in files:
                ext = f.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    res = self.scan_image(str(f), output_dir, save_crops=save_crops)
                    all_results.append(res)
                elif ext in VIDEO_EXTENSIONS:
                    res = self.scan_video(str(f), output_dir, save_annotated_video=save_annotated_video, save_crops=save_crops)
                    all_results.append(res)

        # Save summary report JSON
        report_json = os.path.join(output_dir, "scan_report.json")
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Save CSV summary table
        report_csv = os.path.join(output_dir, "scan_summary.csv")
        self._export_csv(all_results, report_csv)

        print("\n" + "=" * 70)
        print("🎯 SCAN COMPLETED SUCCESSFULLY!")
        print(f"  Processed items   : {len(all_results)}")
        print(f"  Output directory  : {output_dir}")
        print(f"  JSON Report       : {report_json}")
        print(f"  CSV Report        : {report_csv}")
        print("=" * 70)

        return all_results

    def _export_csv(self, results: list, csv_path: str):
        rows = []
        for r in results:
            fname = Path(r["file"]).name
            if "detections" in r:
                # Image result
                if not r["detections"]:
                    rows.append({
                        "File": fname, "Type": "Image", "Status": "Clear",
                        "Threat": "None", "Conf": 0.0, "Laplacian": 0.0,
                        "Time/Frames": "-", "Evidence": "-"
                    })
                else:
                    for d in r["detections"]:
                        rows.append({
                            "File": fname, "Type": "Image", "Status": "ALERT",
                            "Threat": d["class_name"].upper(), "Conf": d["confidence"],
                            "Laplacian": d["laplacian_variance"], "Time/Frames": "-",
                            "Evidence": d.get("crop_path", "-")
                        })
            elif "events" in r:
                # Video result
                if not r["events"]:
                    rows.append({
                        "File": fname, "Type": "Video", "Status": "Clear",
                        "Threat": "None", "Conf": 0.0, "Laplacian": 0.0,
                        "Time/Frames": f"{r.get('total_frames', 0)} frames", "Evidence": "-"
                    })
                else:
                    for ev in r["events"]:
                        rows.append({
                            "File": fname, "Type": "Video", "Status": "RED ALERT",
                            "Threat": f"{ev['class_name'].upper()} (#{ev['track_id']})",
                            "Conf": ev["peak_conf"], "Laplacian": ev["best_laplacian_var"],
                            "Time/Frames": f"{ev['first_seen_time']} - {ev['last_seen_time']}",
                            "Evidence": ev.get("evidence_image", "-")
                        })

        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
