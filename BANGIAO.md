# BÀN GIAO — Hệ thống phát hiện hung khí (PTCHTTM)

Cập nhật: 2026-11-09. Viết cho agent lập trình tiếp nhận công việc.

Ưu tiên của chủ dự án, theo đúng thứ tự: **(1) tỷ lệ bắt được — không bỏ sót,
(2) khoanh vùng chính xác**. Mọi đánh đổi phải nghiêng về hai mục này.

---

## 1. Kiến trúc — đọc trước khi sửa bất cứ gì

Hệ thống có **một lõi phát hiện duy nhất**: `src/common/detector.py`,
class `WeaponDetector`. Trước đây bốn đường chạy mỗi file tự nạp model, tự
tra tên lớp, tự lọc ngưỡng, tự vẽ hộp. Hệ quả là sửa ba chỗ vẫn sót một, và
đó là nguồn của gần như mọi lỗi trong mục 6 dưới đây.

Lõi tách thành ba tầng:

| Hàm | Việc |
|---|---|
| `predict_raw(frame)` | **Một** lần `model.predict()` mỗi khung hình. Trả về `[x1,y1,x2,y2,conf,cls_id]` |
| `classify(raw, w, h, apply_class_threshold)` | Áp T2 (ngưỡng lớp) + T3 (hình học) + T4 (phân mức) |
| `detect(frame)` | Đủ bốn tầng. Dùng cho ảnh tĩnh |
| `detect_for_tracking(frame)` | T1+T3+T4, **bỏ T2**. Dùng cho đường chạy có ByteTrack |
| `make_tracker()` | ByteTrack với tham số đọc từ config |
| `class_name(cls_id)` | Tên lớp, đọc từ chính file `.pt` |

**Vì sao `detect_for_tracking` bỏ T2:** ByteTrack giai đoạn 2 cần detection
mờ (conf thấp) để bám vết. Cắt ngưỡng lớp ngay tầng dò là tự bỏ mất vật thể.
Việc xác nhận theo ngưỡng lớp làm sau, trên vết đã bám.

Bốn file dùng lõi này: `process_image.py`, `process_video.py`,
`src/scanner/media_scanner.py`, `src/pipeline/inference_engine.py`. Cộng
`app.py` (giao diện Gradio). **Đừng thêm `model.predict()` mới ở đâu nữa.**

Nguồn sự thật duy nhất: `configs/system_config.yaml`, đọc qua
`src/common/class_policy.py`. Ba hàm nạp: `load_policy()`,
`load_inference_cfg()`, `load_tracking_cfg()`.

**Tên lớp LUÔN đọc từ file `.pt`.** Không gán cứng bảng tên ở bất kỳ đâu.
`WeaponDetector` raise `RuntimeError` nếu không đọc được — cố ý, vì bản cũ
im lặng dùng bảng tên sai.

---

## 2. Bốn tầng lọc

```
T1  predict() ở ngưỡng sàn thấp = min(detect_conf_floor, ngưỡng lớp nhỏ nhất)
T2  ngưỡng RIÊNG từng lớp (Pistol 0.30 ... Chain 0.60)
T3  hình học — giới hạn diện tích NỘI SUY THEO ĐỘ TIN CẬY
T4  phân mức weapon / suspect, chỉ weapon mới báo động đỏ
```

T3 là phần đáng chú ý nhất. Bản cũ loại thẳng mọi hộp `>12%` diện tích
khung — luật đó cắt cả vật thật ở gần camera, tức là đánh đổi recall để mua
precision, đi ngược yêu cầu của dự án. Bản mới:

```
conf <= conf_lo (0.35)      -> giới hạn CHẶT  12%  (suspect 8%)
conf >= bypass_conf (0.65)  -> giới hạn RỘNG  35%  (suspect 18%)
ở giữa                      -> nội suy tuyến tính
vượt hard_max (65%)         -> loại, không miễn trừ
```

Cùng một cái hộp 23.6% khung: conf 0.88 thì **giữ** (dao giơ gần camera),
conf 0.22 thì **loại** (cánh tay). Căn cứ: chẩn đoán trên video thật cho
thấy nhiễu bám cánh tay/balo nằm ở conf 0.10–0.30.

`keep_geometry()` nhận `conf` và `cls_name` là tham số **tuỳ chọn**, nhưng
**bắt buộc phải truyền**. Thiếu thì hàm trả về mức cho phép RỘNG NHẤT và bộ
lọc coi như không còn tác dụng. Đây từng là một lỗi thật (mục 6).

---

## 3. Trạng thái dataset

`data/weapon_8cls.yaml` → `data/dataset_merged`, 8 lớp:
`Axe, Sickle, Sword, Crowbar, Knife, Pistol, Long_gun, Grenade`
(id 0–7 giữ nguyên từ bộ 12 lớp, nên không cần remap)

Audit gần nhất:

| | train | valid | test |
|---|---|---|---|
| Ảnh | 19 956 | 5 223 | 4 037 |
| Hộp | 20 162 | 5 320 | 5 028 |
| Ảnh nền | 21.5% | 21.3% | 5.1% |
| Trung vị DT hộp | 12.01% | 16.55% | 17.72% |
| Hộp > 25% khung | 32.4% | 38.8% | 40.6% |

**Ba vấn đề còn lại, chưa chữa:**

1. **Hộp vẫn quá to.** Axe trung vị **55.97%** khung, p90 97.19%. Sickle
   47.59%. Y nguyên con số của bộ 12 lớp. Đây là nguyên nhân gốc của lỗi
   nhầm cánh tay/balo, và nó **không phải lỗi code**. `rescale_oversized.py`
   là đòn chữa đã viết sẵn nhưng **chưa chạy** (xem mục 4).

2. **Sword: 2372 hộp nhưng chỉ 261 ảnh** (9.1 hộp/ảnh, các lớp khác ~1.2).
   Đây là ảnh giá trưng bày. Độ đa dạng thật của Sword là 261 cảnh, không
   phải 2372. Tính theo ảnh, mất cân bằng là **20.8×** (Knife 5435 / Sword
   261), không phải 7.2× như tính theo hộp. **Không chữa được bằng code** —
   cần ảnh mới.

3. **Chia tập chưa phân tầng.** Knife: train 31.4% → valid 21.5% → test
   18.3%. Nguyên nhân: `fix_sohas_split.py` chia 85/15 trong khi tỉ lệ gốc
   của dự án là ~77/23, nên train nhận quá nhiều SOHAS (phần lớn là Knife).
   Ảnh hưởng: dò ngưỡng trên valid và chọn epoch tốt nhất bị lệch nhẹ.
   Sửa được: hạ `VALID_PCT` trong `fix_sohas_split.py` và chia lại theo lớp.

Liên quan: Pistol trung vị **15.36% ở train** nhưng **30–32% ở valid/test** —
mô hình học trên súng nhỏ rồi bị đo trên súng to.

**Hệ quả cần biết trước khi đọc số:** `evaluate.py` dùng `model.val()`,
không đi qua bộ lọc hình học; `app.py` thì có. Với Axe trung vị 56% khung,
hai đường sẽ cho **hai con số khác nhau**. Không phải lỗi — đúng thiết kế,
vì miền triển khai là CCTV nơi vũ khí nhỏ. Đừng "sửa" cho hai số bằng nhau.

---

## 4. Việc còn lại, theo thứ tự

```
# 1. Thu nhỏ ảnh có hộp quá to, dán vào ảnh nền thật  <-- CHƯA CHẠY
python rescale_oversized.py                  # xem trước
python rescale_oversized.py --apply          # ~10-20 phút, thêm ~25% ảnh train

# 2. Kiểm lại: trung vị DT hộp phải GIẢM RÕ RỆT
python audit_dataset.py --data data/weapon_8cls.yaml

# 3. Sao lưu trọng số cũ TRƯỚC KHI TRAIN (mAP50 0.8504, mất là phải train lại)
copy weights\best.pt weights\best_12cls_old.pt

# 4. Train
python train_full.py

# 5. Đo và chốt ngưỡng
python src/training/evaluate.py --split test
python src/training/evaluate.py --tune-thresholds --beta 2.0
#   -> dán bảng YAML nó in ra vào class_policy.thresholds
#   beta=2.0 là cố ý: ưu tiên giảm bỏ sót
```

Bảng ngưỡng trong config hiện tại là **số phỏng đoán**, chưa phải số đo.
Bước 5 là bước thay nó bằng số thật.

Tuỳ chọn, nếu còn thời gian: sửa vấn đề 3 ở mục trên (chia tập phân tầng).

---

## 5. ĐỪNG "SỬA" NHỮNG THỨ NÀY

Bốn chỗ trông như lỗi nhưng không phải. Đã có người (tôi) báo động nhầm một
trong số này rồi.

1. **ByteTrack dùng `cost = 1 - IoU` và điều kiện `cost <= match_thresh`.**
   Đúng ngữ nghĩa gốc. `match_thresh = 0.80` tương đương `IoU >= 0.20`.
   Đừng đảo lại thành dạng trông "hợp lý" hơn.

2. **`iou_threshold: 0.70`, không phải 0.45.** Config ban đầu ghi 0.45 nhưng
   `detector.py` chưa bao giờ truyền vào `predict()` — code chết, hệ thống
   chạy mặc định 0.7 của ultralytics. Nay nối thật và **giữ 0.70**: NMS với
   iou thấp xoá bớt hộp chồng nhau, hai người đứng cạnh nhau mỗi người một
   con dao sẽ bị bỏ mất một. Hạ xuống 0.45 là **giảm** recall.

3. **`degrees / shear / perspective = 0`.** Cố ý, không phải bỏ quên. Quay
   ảnh làm hộp bao song song trục phình to ra: một con dao mảnh quay 30° có
   hộp rộng gần gấp đôi vật thật. Nhãn thành sai, mà hộp lỏng đúng là cái
   đang cần chữa.

4. **`mixup = 0.0`.** Mixup trộn hai ảnh trong suốt và giữ hộp của **cả
   hai**, buộc mô hình đoán hộp cho vật bị chồng mờ → biên hộp học được nhoè
   đi. Lợi cho phân lớp, hại cho độ sát của hộp. Ưu tiên khoanh vùng thì
   phải tắt.

---

## 6. Lịch sử lỗi đã sửa — đừng tái tạo

| Lỗi | Hậu quả | Nằm ở |
|---|---|---|
| Bảng tên lớp gán cứng 3 lớp `{gun, knife, bat_stick}` | 3 tên của mô hình cũ, mô hình nay 8 lớp → mọi lớp rơi vào nhánh dự phòng, không hiện tên tiếng Việt | `inference_engine.py`, `media_scanner.py` |
| Bảng tên lớp gán cứng 12 lớp | mô hình nay 8 lớp → lệch id là sai tên | `process_image.py`, `process_video.py` |
| `conf=0.30` gán cứng trong `predict()` | Pistol/Long_gun/Grenade ngưỡng riêng đúng 0.30 → bị chặn ngay tầng dò, ByteTrack giai đoạn 2 không có dữ liệu | `media_scanner.py` |
| `keep_geometry()` gọi thiếu `conf` | hàm trả mức cho phép rộng nhất → bộ lọc mất tác dụng hoàn toàn | `process_image.py`, `process_video.py` |
| Không gọi `keep_geometry()` bao giờ | camera trực tiếp và quét media **không có** tầng lọc hình học nào | `inference_engine.py`, `media_scanner.py` |
| `self.half` tính rồi không truyền vào `predict()` | biến chết, FP16 không có tác dụng | `inference_engine.py` |
| `WeaponByteTracker(track_high_thresh=0.45, ...)` gán cứng, **bỏ quên `new_track_thresh`** | bản sửa `new_track_thresh=0.35` chỉ có tác dụng ở pipeline camera; đường quét video vẫn dùng mặc định → vũ khí di chuyển ở conf trung bình không mở được track mới | `process_video.py` |
| `process_sohas.py` lặp `[('test','test')]` | 857 ảnh SOHAS đổ vào tập **test** → mốc mAP 0.8504 mất khả năng so sánh, và dữ liệu đúng miền nhất nằm ở test không train được | `process_sohas.py` |
| Manifest lưu **đường dẫn tuyệt đối** | đổi tên/di chuyển thư mục dự án → `--undo` xoá file ở thư mục **cũ**. Đã xảy ra thật trong lúc test | `fix_sohas_split.py`, `rescale_oversized.py` |
| Manifest bị **ghi đè** mỗi lần apply | apply lần hai (không tạo gì) xoá sạch bản ghi lần một → `--undo` vô dụng | `rescale_oversized.py` |
| `min(self.conf_threshold, ...)` với `conf_threshold=None` | `TypeError` ngay lúc khởi tạo luồng | `inference_engine.py` |

Hai lỗi cuối và lỗi `keep_geometry` thiếu `conf` là lỗi **do chính đợt
refactor này tạo ra rồi sửa**. Mỗi script có một bài kiểm thử canh đúng chỗ
đó; đừng bỏ các bài đó khi sửa tiếp.

---

## 7. Ba script quản lý dataset

Cả ba đều theo cùng một khuôn: **mặc định là xem trước, không ghi gì**;
`--apply` để thực hiện; `--undo` để hoàn tác theo manifest. Manifest lưu
đường dẫn **tương đối** so với thư mục dự án và có hàm `unrel()` chặn mọi
đường trỏ ra ngoài dự án.

- `rebalance_dataset.py` — bỏ 4 lớp id 8–11 (Stick, Hammer, Spanner, Chain).
  **Đã chạy.** Sao lưu ở `labels_backup_12cls/`.
- `fix_sohas_split.py` — chuyển dữ liệu SOHAS ra khỏi tập test. **Đã chạy.**
- `rescale_oversized.py` — **CHƯA CHẠY.** Xem mục 4.

### Về `rescale_oversized.py`

Thu nhỏ ảnh có hộp lớn nhất vượt 25% khung, dán vào một ảnh nền thật lấy từ
chính kho 4287 hard negative đang có, hộp lớn nhất còn 3–12% khung. Ảnh gốc
**được giữ**, nên mô hình thấy cả hai cỡ.

**Phần quan trọng, đừng bỏ:** dán ảnh chữ nhật lên ảnh khác để lại **đường
ghép**. Nếu đường ghép chỉ xuất hiện ở ảnh **có** vũ khí thì mô hình học
"thấy đường ghép là có vũ khí" — một đặc trưng giả, còn tệ hơn vấn đề ban
đầu. Hai biện pháp, script dùng cả hai:

- làm mờ biên dán (`feather_paste`) + cân độ sáng (`match_brightness`), nên
  không còn mép cứng và không còn ô sáng/tối rõ rệt;
- sinh thêm 25% **ảnh ghép ÂM**: một ảnh nền thu nhỏ dán vào ảnh nền khác,
  nhãn **rỗng**. Đường ghép khi đó xuất hiện ở cả ảnh có và không có vũ khí
  nên không mang thông tin gì.

Nếu sửa script này, giữ nguyên cả hai biện pháp.

Script chỉ chạm **tập train**. Augment valid/test là tự lừa mình.

---

## 8. Cách kiểm thử mà không cần GPU

Không có GPU hay file `.pt` thật vẫn kiểm được phần lớn logic, bằng cách
stub `ultralytics` và `torch`:

```python
# đặt trong sitecustomize.py hoặc import trước module cần test
class YOLO:
    def __init__(self, w):
        self.names = {0:"Axe", 4:"Knife", 5:"Pistol", ...}
    def predict(self, frame, **kw):
        CALLS.append(kw)          # để assert iou/max_det/half có được truyền
        return [FakeResult([...])]  # box với conf và cls_id tự đặt
```

Bộ ca kiểm thử tối thiểu nên giữ:

1. `class_policy.py` chạy trực tiếp (`python src/common/class_policy.py`) in
   ra bảng tự kiểm 6 ca hình học. Ca quan trọng nhất: **cùng một hộp 23.6%
   khung, conf 0.88 phải GIỮ, conf 0.22 phải LOẠI.**
2. `predict()` nhận đúng `imgsz=960, iou=0.70, max_det=100, half`.
3. `process_image` và `media_scanner` trả **cùng số** phát hiện trên cùng
   input — nếu lệch thì hai đường chạy đã tách logic trở lại.
4. `detect_for_tracking()` cho detection **dưới** ngưỡng lớp đi qua, nhưng
   **chặn** hộp cỡ cánh tay ở conf thấp.
5. `make_tracker().new_track_thresh == 0.35` (đọc từ config, không gán cứng).
6. `AIInferenceTrackingEngine(...)` khởi tạo được với `conf_threshold=None`.
7. Mỗi script dataset: vòng `--apply` → `--undo` phải cho **MD5 khớp hoàn
   toàn** với trạng thái trước đó.

---

## 9. Dataset nguồn

- Bộ hiện dùng: Roboflow 12 lớp đã rebalance xuống 8, trộn thêm SOHAS.
- SOHAS: `https://github.com/ari-dasci/OD-WeaponDetection`, thư mục
  `Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/`.
  Đã sẵn định dạng YOLO. Giấy phép ghi khác nhau giữa thân trang (CC BY-SA
  4.0) và sidebar GitHub (CC-BY-4.0) — đọc `License.md` trước khi trích dẫn
  trong báo cáo.
- Chi tiết đầy đủ, kèm số đo và việc cần làm với SOHAS: `CLONE_DATASET.txt`.

**Đòn mạnh nhất còn lại, không script nào thay được:** gán nhãn 200–500
khung từ video demo của chính chủ dự án. Đúng miền, đúng góc camera, đúng độ
phân giải mà hệ thống sẽ chạy. Một mình bước đó có tác dụng hơn mọi thứ
trong mục 4.
