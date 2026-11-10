# Hệ Thống Trinh Sát Phát Hiện Hung Khí (Weapon Detection System)
**Học viện Công nghệ Bưu chính Viễn thông (PTIT) - PTCHTTM**

Hệ thống trinh sát và cảnh báo hung khí nguy hiểm tự động thời gian thực sử dụng thị giác máy tính, mô hình **YOLO26s**, thuật toán bám vết **ByteTrack** và giao diện điều khiển **Java Swing**.

---

## 🌟 Tính Năng Nổi Bật

1. **📷 Giám Sát Camera Trực Tiếp (Live CCTV / Webcam)**:
   - Truyền luồng video trực tiếp 30 FPS ngay trong thẻ giao diện Java Swing.
   - Khoanh vùng bám sát đối tượng bằng hình elip động và 4 góc ngắm radar bám sát hộp bao.
   - Khi phát hiện hung khí: Viền màn hình chuyển sang **Cảnh báo đỏ (Crimson Red Border)** rực rỡ và phát tín hiệu báo động khẩn cấp.
   - Hỗ trợ chọn linh hoạt giữa Webcam mặc định (cam 0), Webcam ngoài (cam 1), hoặc Chế độ video an ninh mô phỏng.

2. **🎬 Xử Lý Video (Video Processing & Tracking)**:
   - Quét từng khung hình của video với thuật toán bám vết **ByteTrack**.
   - Khử rung lắc và lọc hình học đa tầng (loại bỏ bóng râm, mép tường, cánh tay, balo).
   - Hiển thị thanh tiến độ thời gian thực (FPS, % hoàn thành) và mở video thành phẩm ngay trên máy.

3. **🖼️ Xử Lý Hình Ảnh (Image Scanning & Labeling)**:
   - Quét phát hiện vũ khí tức thì trên ảnh tĩnh.
   - Hiển thị song song ảnh gốc và ảnh đã khoanh vùng gắn nhãn tiếng Việt rõ nét.
   - Thống kê chi tiết danh sách hung khí, độ tin cậy và tọa độ.

---

## 🎯 8 Lớp Vũ Khí Được Hỗ Trợ

Hệ thống tập trung vào 8 loại vũ khí thật nguy hiểm, phân tầng cảnh báo:
* **🔴 Nhóm Vũ Khí Nguy Hiểm**:
  * `Súng ngắn (Pistol)`
  * `Súng trường / Súng dài (Long gun)`
  * `Dao các loại (Knife)`
  * `Kiếm / Mã tấu (Sword)`
  * `Rìu chặt (Axe)`
  * `Lưỡi liềm / Rựa (Sickle)`
  * `Lựu đạn (Grenade)`
* **🟡 Nhóm Dụng Cụ / Nghi Vấn**:
  * `Xà beng (Crowbar)`

---

## 📊 Hiệu Năng Mô Hình (Validation trên 5.223 ảnh độc lập)

* **mAP@50**: **88.9% (~89%)**
* **Precision**: **86.9%** (hạn chế tối đa báo sai, nhận nhầm)
* **Recall**: **82.3%** (ưu tiên phát hiện triệt để, không bỏ sót hung khí)
* **Thời gian suy luận**: ~8ms / frame trên GPU NVIDIA RTX 4050.

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu cầu môi trường
* **Hệ điều hành**: Windows 10/11
* **Java**: JDK 17 trở lên (đã kiểm thử tương thích với OpenJDK 21)
* **Python**: 3.10 - 3.13 (khuyên dùng Python 3.13)
* **GPU (Tùy chọn)**: NVIDIA GPU hỗ trợ CUDA để đạt hiệu năng tối đa (hoặc CPU).

### 2. Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

### 3. Biên dịch và Chạy Giao Diện Java Swing
Biên dịch:
```bash
javac -encoding UTF-8 WeaponDetectionPanel.java
```

Chạy chương trình:
```bash
java WeaponDetectionPanel
```

---

## 📁 Cấu Trúc Dự Án

```
c:/ptchttm/
├── configs/
│   └── system_config.yaml       # Cấu hình chính sách ngưỡng, lọc hình học và NMS
├── src/
│   ├── common/
│   │   ├── detector.py          # Lõi nhận diện YOLO26s & vẽ radar HUD
│   │   └── class_policy.py      # Chính sách ngưỡng riêng từng lớp & phân mức
│   ├── tracking/
│   │   └── weapon_tracker.py    # Thuật toán bám vết ByteTrack
│   └── agent/                   # Module AI phân tích an ninh
├── weights/
│   └── best.pt                  # Trọng số mô hình tối ưu đã huấn luyện
├── swing_bridge.py              # Cầu nối IPC siêu tốc giữa Python và Java Swing
├── WeaponDetectionPanel.java    # Giao diện thẻ Java Swing đầy đủ 3 chức năng
├── process_image.py             # CLI xử lý ảnh độc lập
├── process_video.py             # CLI xử lý video độc lập
└── requirements.txt             # Danh sách thư viện Python
```

---

## 👥 Tác Giả & Bản Quyền
* **Đơn vị thực hiện**: Học viện Công nghệ Bưu chính Viễn thông (PTIT).
* **Môn học**: Phát triển các hệ thống thông minh (PTCHTTM).
