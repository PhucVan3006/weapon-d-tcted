"""
Multi-Object Weapon Tracking with ByteTrack & Sliding-Window Dual Thresholding.

Theoretical & Engineering Basis:
- Kalman-style motion model: predicts state of the weapon box across frames.
- Hungarian Algorithm: 2-stage bipartite matching on an IoU cost matrix.
    * Stage 1: high-confidence boxes (conf >= track_high_thresh)
    * Stage 2: low-confidence boxes (track_low_thresh <= conf < track_high_thresh)
      Prevents track fragmentation when a knife/gun is blurred by fast motion.
- Sliding Window + Dual-Thresholding verification before raising a RED ALERT.

-----------------------------------------------------------------------------
SỬA LỖI v2 - ba thay đổi, tất cả đều nhằm GIẢM BỎ SÓT
-----------------------------------------------------------------------------
1. predict() không còn bị gọi nhiều lần mỗi frame.
   Bản cũ gọi predict() một lần trong update(), rồi _match() lại gọi
   predict() thêm lần nữa cho mỗi track, mà _match() chạy hai giai đoạn
   -> mỗi track bị "tua" tới 3 lần mỗi frame. Vì predict() cộng dồn
   `xc += vx`, hộp dự đoán bay xa gấp 3 lần vận tốc thật, IoU với hộp
   thật tụt mạnh, track đứt gãy đúng lúc hung khí đang di chuyển.
   Nay predict() chạy đúng MỘT lần/frame và lưu kết quả vào
   `last_pred_box`; _match() chỉ đọc lại qua peek_box().

2. Track mới sinh được từ detection độ tin cậy trung bình.
   Bản cũ chỉ tạo track từ detection conf >= track_high_thresh (0.50).
   Một con dao ở xa luôn cho conf 0.35-0.45 thì không bao giờ tạo nổi
   track -> không vào cửa sổ trượt -> không bao giờ báo động. Nay dùng
   `new_track_thresh` (mặc định 0.35) cho việc khởi tạo track.

3. Luật báo động chuyển sang dạng HOẶC.
   Bản cũ bắt buộc phải có ít nhất một frame conf >= 0.55. Vật thể
   xuất hiện ổn định 8/8 frame nhưng conf đỉnh 0.52 vẫn bị bỏ qua.
   Nay: (đủ hits VÀ có đỉnh conf cao) HOẶC (xuất hiện rất dày đặc).

GHI CHÚ VỀ NGƯỠNG KHỚP - KHÔNG PHẢI LỖI, ĐỪNG "SỬA":
   cost = 1 - IoU, nên điều kiện `cost <= match_thresh` với
   match_thresh=0.8 tương đương IoU >= 0.2. Đây đúng là ngữ nghĩa của
   ByteTrack gốc (linear_assignment(dists, thresh=0.8)). Giai đoạn 2
   dùng 0.5 (IoU >= 0.5) chặt hơn cũng đúng bản gốc: detection độ tin
   cậy thấp cần khớp vị trí tốt hơn mới được chấp nhận.
"""

import numpy as np
from collections import deque, Counter
from scipy.optimize import linear_sum_assignment


def bbox_iou(box1, box2):
    """Computes IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    b2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class KalmanBoxTracker:
    """
    Motion tracker for a bounding box [x1, y1, x2, y2].
    State: [xc, yc, s, r] + velocities [vx, vy, vs], where s = area, r = w/h.
    """
    count = 0

    def __init__(self, bbox, cls_id=0, conf=0.5, window_size=8):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.xc = bbox[0] + w / 2.0
        self.yc = bbox[1] + h / 2.0
        self.s = w * h
        self.r = w / float(h) if h > 0 else 1.0

        self.vx = 0.0
        self.vy = 0.0
        self.vs = 0.0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.cls_id = cls_id
        self.conf = conf
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

        # Cửa sổ trượt: (hit_flag, conf, bbox) của N frame gần nhất
        self.sliding_window = deque(maxlen=window_size)
        self.sliding_window.append((True, conf, bbox))

        # Lịch sử nhãn để bỏ phiếu đa số, chống đổi nhãn giật cục giữa các frame
        self.cls_history = deque(maxlen=window_size)
        self.cls_history.append(cls_id)

        self.alert_triggered = False
        self.best_shot_score = -1.0
        self.best_shot_img = None
        self.best_shot_box = bbox

        # Hộp dự đoán của frame hiện tại. _match() đọc lại giá trị này
        # thay vì gọi predict() thêm lần nữa.
        self.last_pred_box = list(bbox)

    # -- Motion model ----------------------------------------------------
    def _box_from_state(self):
        w = np.sqrt(max(0.0, self.s * self.r))
        h = self.s / w if w > 0 else 0.0
        return [self.xc - w / 2.0, self.yc - h / 2.0,
                self.xc + w / 2.0, self.yc + h / 2.0]

    def predict(self):
        """
        Tiến một bước theo mô hình chuyển động.
        CHỈ ĐƯỢC GỌI MỘT LẦN cho mỗi track mỗi frame.
        """
        self.xc += self.vx
        self.yc += self.vy
        self.s = max(1.0, self.s + self.vs)
        self.age += 1
        self.time_since_update += 1

        self.last_pred_box = self._box_from_state()
        return self.last_pred_box

    def peek_box(self):
        """Đọc hộp đã dự đoán mà KHÔNG làm thay đổi trạng thái."""
        return self.last_pred_box

    def update(self, bbox, conf, cls_id):
        """Cập nhật trạng thái bằng hộp quan sát được."""
        self.time_since_update = 0
        self.hits += 1
        self.conf = conf

        # Bỏ phiếu đa số theo lịch sử gần nhất để chống rung giật nhãn
        self.cls_history.append(cls_id)
        counts = Counter(self.cls_history)
        most_common_cls, count = counts.most_common(1)[0]
        if count >= (len(self.cls_history) + 1) // 2:
            self.cls_id = most_common_cls

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        new_xc = bbox[0] + w / 2.0
        new_yc = bbox[1] + h / 2.0
        new_s = w * h
        new_r = w / float(h) if h > 0 else 1.0

        alpha = 0.7
        self.vx = alpha * (new_xc - self.xc) + (1 - alpha) * self.vx
        self.vy = alpha * (new_yc - self.yc) + (1 - alpha) * self.vy
        self.vs = alpha * (new_s - self.s) + (1 - alpha) * self.vs

        self.xc = new_xc
        self.yc = new_yc
        self.s = new_s
        self.r = new_r
        self.last_pred_box = self._box_from_state()

        self.sliding_window.append((True, conf, bbox))

    def mark_missed(self):
        """Gọi khi không có detection nào khớp track này trong frame hiện tại."""
        self.sliding_window.append((False, 0.0, None))

    def get_current_box(self):
        b = self._box_from_state()
        return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

    # -- Alert gate ------------------------------------------------------
    def evaluate_red_alert(self, min_hits=4, min_peak_conf=0.55, strong_hits=6):
        """
        Luật xác nhận RED ALERT (dạng HOẶC):

          (A) Xuất hiện >= min_hits frame trong cửa sổ
              VÀ có ít nhất 1 frame conf >= min_peak_conf
          HOẶC
          (B) Xuất hiện >= strong_hits frame trong cửa sổ

        Nhánh (B) là phần bổ sung để giảm bỏ sót: một con dao ở xa có thể
        không bao giờ đạt conf 0.55, nhưng nếu nó hiện diện 6/8 frame
        thì đó không còn là nhiễu ngẫu nhiên.
        """
        if len(self.sliding_window) < min_hits:
            return False

        hits_count = sum(1 for hit, _, _ in self.sliding_window if hit)
        peak_conf = max((conf for hit, conf, _ in self.sliding_window if hit),
                        default=0.0)

        cond_a = (hits_count >= min_hits) and (peak_conf >= min_peak_conf)
        cond_b = hits_count >= strong_hits
        return cond_a or cond_b


class WeaponByteTracker:
    def __init__(
        self,
        track_high_thresh=0.50,
        track_low_thresh=0.15,
        track_buffer=30,
        match_thresh=0.80,
        new_track_thresh=0.35,
        window_size=8,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        # Ngưỡng tối thiểu để một detection chưa khớp được phép MỞ track mới.
        self.new_track_thresh = new_track_thresh
        self.window_size = window_size
        self.tracks = []

    def update(self, detections):
        # --- B1: tiến mô hình chuyển động, ĐÚNG MỘT LẦN mỗi frame -------
        for t in self.tracks:
            t.predict()

        dets_high = [d for d in detections if d[4] >= self.track_high_thresh]
        dets_low = [d for d in detections
                    if self.track_low_thresh <= d[4] < self.track_high_thresh]

        # --- B2: khớp giai đoạn 1 với detection độ tin cậy cao ----------
        unmatched_tracks, matched_high_pairs, unmatched_high_dets = self._match(
            self.tracks, dets_high, cost_thresh=self.match_thresh
        )

        for track_idx, det_idx in matched_high_pairs:
            d = dets_high[det_idx]
            self.tracks[track_idx].update(bbox=d[:4], conf=d[4], cls_id=int(d[5]))

        # --- B3: khớp giai đoạn 2 với detection độ tin cậy thấp ---------
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        unmatched_tracks_stage2, matched_low_pairs, unmatched_low_dets = self._match(
            remaining_tracks, dets_low, cost_thresh=0.5
        )

        for rel_track_idx, det_idx in matched_low_pairs:
            orig_track = remaining_tracks[rel_track_idx]
            d = dets_low[det_idx]
            orig_track.update(bbox=d[:4], conf=d[4], cls_id=int(d[5]))

        truly_unmatched = [remaining_tracks[i] for i in unmatched_tracks_stage2]
        for t in truly_unmatched:
            t.mark_missed()

        # --- B4: mở track mới -------------------------------------------
        # Mọi detection chưa khớp và đạt new_track_thresh đều được mở track,
        # kể cả loại độ tin cậy trung bình (0.35-0.50). Đây là thay đổi
        # quan trọng nhất để giảm bỏ sót vật thể nhỏ / ở xa.
        new_det_pool = [dets_high[i] for i in unmatched_high_dets]
        new_det_pool += [dets_low[i] for i in unmatched_low_dets]

        for d in new_det_pool:
            if d[4] >= self.new_track_thresh:
                self.tracks.append(KalmanBoxTracker(
                    bbox=d[:4], cls_id=int(d[5]), conf=d[4],
                    window_size=self.window_size
                ))

        # --- B5: loại bỏ track quá hạn ----------------------------------
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.track_buffer]
        return self.tracks

    def _match(self, tracks, detections, cost_thresh=0.8):
        """
        Khớp track với detection bằng Hungarian trên ma trận chi phí.

        cost = 1 - IoU, nên `cost <= cost_thresh` nghĩa là
        `IoU >= 1 - cost_thresh`. Giữ nguyên ngữ nghĩa ByteTrack gốc.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return list(range(len(tracks))), [], list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            # Đọc hộp đã dự đoán ở B1 - KHÔNG gọi lại predict() ở đây.
            t_box = t.peek_box()
            for j, d in enumerate(detections):
                cost_matrix[i, j] = 1.0 - bbox_iou(t_box, d[:4])

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= cost_thresh:
                matched_pairs.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return sorted(unmatched_tracks), matched_pairs, sorted(unmatched_dets)
