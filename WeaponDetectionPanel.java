import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.border.TitledBorder;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * WeaponDetectionPanel - Thẻ (Tab / Panel) Java Swing Phát hiện Hung khí.
 * Học viện Công nghệ Bưu chính Viễn thông (PTIT) - PTCHTTM.
 * 
 * Có 3 chức năng:
 *   1. Camera trực tiếp (Live Camera CCTV/Webcam)
 *   2. Xử lý video (Video Processing & Bounding)
 *   3. Xử lý hình ảnh (Image Scanning & Labeling)
 * 
 * Khi phát hiện hung khí: Tự động khoanh vùng, gắn nhãn tiếng Việt và bật CẢNH BÁO ĐỎ.
 */
public class WeaponDetectionPanel extends JPanel {

    private static final File WORK_DIR = new File("C:/ptchttm").exists()
            ? new File("C:/ptchttm")
            : new File(".").getAbsoluteFile();

    private static String getPythonExe() {
        File py313 = new File("C:/Python313/python.exe");
        if (py313.exists()) {
            return py313.getAbsolutePath();
        }
        return "python";
    }

    private static String getBridgeScript() {
        File f = new File(WORK_DIR, "swing_bridge.py");
        if (f.exists()) return f.getAbsolutePath();
        return "swing_bridge.py";
    }

    // Helper trích xuất JSON bằng Regular Expression an toàn
    private static String extractJsonString(String json, String key) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return m.group(1).replace("\\\\", "\\").replace("\\/", "/");
        }
        return null;
    }

    private static int extractJsonInt(String json, String key) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("\"" + key + "\"\\s*:\\s*(\\d+)");
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            try {
                return Integer.parseInt(m.group(1));
            } catch (Exception ignored) {}
        }
        return 0;
    }

    private static boolean extractJsonBool(String json, String key) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("\"" + key + "\"\\s*:\\s*(true|false)");
        java.util.regex.Matcher m = p.matcher(json);
        if (m.find()) {
            return Boolean.parseBoolean(m.group(1));
        }
        return false;
    }

    // Components Tab Ảnh
    private JLabel lblOriginalImage;
    private JLabel lblProcessedImage;
    private JLabel lblImageAlertBadge;
    private JTextArea txtImageDetails;
    private File selectedImageFile;

    // Components Tab Video
    private JLabel lblVideoAlertBadge;
    private JTextArea txtVideoDetails;
    private JProgressBar videoProgressBar;
    private File selectedVideoFile;
    private File resultVideoFile;
    private JButton btnOpenResultVideo;

    // Components Tab Camera
    private JLabel lblCameraAlertBadge;
    private JLabel lblCameraScreen;
    private JLabel lblCameraInfo;
    private JComboBox<String> comboCameraSource;
    private Process cameraProcess;
    private volatile boolean isCameraRunning = false;

    public WeaponDetectionPanel() {
        setLayout(new BorderLayout());
        setBorder(new EmptyBorder(10, 10, 10, 10));

        // Header
        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBackground(new Color(25, 30, 45));
        headerPanel.setBorder(new EmptyBorder(12, 16, 12, 16));

        JLabel lblTitle = new JLabel("HỆ THỐNG TRINH SÁT PHÁT HIỆN HUNG KHÍ (YOLO26s)");
        lblTitle.setFont(new Font("Segoe UI", Font.BOLD, 18));
        lblTitle.setForeground(Color.WHITE);

        JLabel lblSub = new JLabel("Học viện Công nghệ Bưu chính Viễn thông · PTCHTTM");
        lblSub.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        lblSub.setForeground(new Color(180, 195, 215));

        headerPanel.add(lblTitle, BorderLayout.NORTH);
        headerPanel.add(lblSub, BorderLayout.SOUTH);
        add(headerPanel, BorderLayout.NORTH);

        // JTabbedPane chứa 3 chức năng (3 Thẻ con)
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.setFont(new Font("Segoe UI", Font.BOLD, 13));

        tabbedPane.addTab("📷 Camera Trực Tiếp", createCameraTab());
        tabbedPane.addTab("🎬 Xử Lý Video", createVideoTab());
        tabbedPane.addTab("🖼️ Xử Lý Hình Ảnh", createImageTab());

        add(tabbedPane, BorderLayout.CENTER);
    }

    // =========================================================================
    // 1. THẺ: XỬ LÝ HÌNH ẢNH
    // =========================================================================
    private JPanel createImageTab() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(new EmptyBorder(10, 10, 10, 10));

        // Thanh công cụ trên
        JPanel toolBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 5));
        JButton btnSelectImage = new JButton("📂 Chọn Ảnh...");
        JButton btnScanImage = new JButton("🔍 Quét & Khoanh Vùng");
        btnScanImage.setEnabled(false);

        JComboBox<String> comboImageResolution = new JComboBox<>(new String[]{
            "⚡ Chuẩn 960px (Khuyến nghị)",
            "🔍 Quét sâu 1280px (Pháp chứng)",
            "🚀 Nhanh 640px"
        });
        comboImageResolution.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        btnSelectImage.setFont(new Font("Segoe UI", Font.BOLD, 12));
        btnScanImage.setFont(new Font("Segoe UI", Font.BOLD, 12));
        btnScanImage.setBackground(new Color(220, 53, 69));
        btnScanImage.setForeground(Color.WHITE);

        lblImageAlertBadge = new JLabel("Trạng thái: Chưa chọn ảnh");
        lblImageAlertBadge.setFont(new Font("Segoe UI", Font.BOLD, 13));
        lblImageAlertBadge.setOpaque(true);
        lblImageAlertBadge.setBackground(new Color(230, 230, 230));
        lblImageAlertBadge.setBorder(new EmptyBorder(5, 12, 5, 12));

        toolBar.add(btnSelectImage);
        toolBar.add(btnScanImage);
        toolBar.add(new JLabel("Độ phân giải:"));
        toolBar.add(comboImageResolution);
        toolBar.add(lblImageAlertBadge);
        panel.add(toolBar, BorderLayout.NORTH);

        // Khung hiển thị ảnh trước / sau
        JPanel viewPanel = new JPanel(new GridLayout(1, 2, 10, 10));

        JPanel leftPanel = new JPanel(new BorderLayout());
        leftPanel.setBorder(new TitledBorder("Ảnh Gốc"));
        lblOriginalImage = new JLabel("Chưa có ảnh", SwingConstants.CENTER);
        lblOriginalImage.setBackground(Color.DARK_GRAY);
        leftPanel.add(new JScrollPane(lblOriginalImage), BorderLayout.CENTER);

        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.setBorder(new TitledBorder("Kết Quả Khoanh Vùng & Gắn Nhãn"));
        lblProcessedImage = new JLabel("Kết quả sẽ hiển thị ở đây", SwingConstants.CENTER);
        rightPanel.add(new JScrollPane(lblProcessedImage), BorderLayout.CENTER);

        viewPanel.add(leftPanel);
        viewPanel.add(rightPanel);
        panel.add(viewPanel, BorderLayout.CENTER);

        // Chi tiết văn bản phía dưới
        txtImageDetails = new JTextArea(4, 50);
        txtImageDetails.setEditable(false);
        txtImageDetails.setFont(new Font("Consolas", Font.PLAIN, 12));
        txtImageDetails.setText("Vui lòng bấm 'Chọn Ảnh' để bắt đầu kiểm tra hung khí...");
        panel.add(new JScrollPane(txtImageDetails), BorderLayout.SOUTH);

        // Sự kiện
        btnSelectImage.addActionListener((ActionEvent e) -> {
            JFileChooser chooser = new JFileChooser();
            File defaultDir = new File(WORK_DIR, "data/dataset_merged/valid/images");
            if (!defaultDir.exists()) defaultDir = WORK_DIR;
            chooser.setCurrentDirectory(defaultDir);

            int res = chooser.showOpenDialog(this);
            if (res == JFileChooser.APPROVE_OPTION) {
                selectedImageFile = chooser.getSelectedFile();
                displayScaledImage(lblOriginalImage, selectedImageFile.getAbsolutePath());
                lblProcessedImage.setIcon(null);
                lblProcessedImage.setText("Bấm 'Quét & Khoanh Vùng' để thực hiện");
                btnScanImage.setEnabled(true);
                lblImageAlertBadge.setText("Sẵn sàng quét: " + selectedImageFile.getName());
                lblImageAlertBadge.setBackground(new Color(220, 240, 255));
                lblImageAlertBadge.setForeground(Color.BLACK);
            }
        });

        btnScanImage.addActionListener((ActionEvent e) -> {
            if (selectedImageFile == null) return;
            btnScanImage.setEnabled(false);
            lblImageAlertBadge.setText("⏳ Đang nhận diện hung khí với YOLO26s...");
            lblImageAlertBadge.setBackground(Color.YELLOW);
            lblImageAlertBadge.setForeground(Color.BLACK);
            txtImageDetails.setText("Đang chạy mô hình YOLO26s nhận diện: " + selectedImageFile.getName() + "...");

            int resIdx = comboImageResolution.getSelectedIndex();
            String imgszVal = (resIdx == 1) ? "1280" : (resIdx == 2) ? "640" : "960";

            SwingWorker<String, Void> worker = new SwingWorker<>() {
                @Override
                protected String doInBackground() throws Exception {
                    ProcessBuilder pb = new ProcessBuilder(
                            getPythonExe(), getBridgeScript(),
                            "--action", "image",
                            "--input", selectedImageFile.getAbsolutePath(),
                            "--imgsz", imgszVal
                    );
                    pb.directory(WORK_DIR);
                    pb.environment().put("PYTHONIOENCODING", "utf-8");
                    pb.environment().put("PYTHONUTF8", "1");
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    StringBuilder fullLog = new StringBuilder();
                    String lastJson = null;

                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            fullLog.append(line).append("\n");
                            String trimmed = line.trim();
                            if (trimmed.startsWith("{") && trimmed.contains("\"status\"")) {
                                lastJson = trimmed;
                            }
                        }
                    }
                    int exitCode = process.waitFor();
                    if (lastJson != null) {
                        return lastJson;
                    }
                    throw new RuntimeException("Lỗi thực thi (Exit code " + exitCode + "):\n" + fullLog);
                }

                @Override
                protected void process(java.util.List<Void> chunks) {}

                @Override
                protected void done() {
                    btnScanImage.setEnabled(true);
                    try {
                        String jsonStr = get();
                        handleImageResult(jsonStr);
                    } catch (Exception ex) {
                        lblImageAlertBadge.setText("❌ Lỗi xử lý: " + ex.getMessage());
                        lblImageAlertBadge.setBackground(Color.PINK);
                        lblImageAlertBadge.setForeground(Color.BLACK);
                        txtImageDetails.setText("LỖI XỬ LÝ ẢNH:\n" + ex.getMessage());
                    }
                }
            };
            worker.execute();
        });

        return panel;
    }

    private void handleImageResult(String jsonStr) {
        if (jsonStr == null || jsonStr.isEmpty()) {
            lblImageAlertBadge.setText("❌ Không nhận được kết quả từ mô hình.");
            return;
        }

        String status = extractJsonString(jsonStr, "status");
        if ("error".equals(status)) {
            String msg = extractJsonString(jsonStr, "message");
            lblImageAlertBadge.setText("❌ Lỗi: " + (msg != null ? msg : jsonStr));
            lblImageAlertBadge.setBackground(Color.PINK);
            lblImageAlertBadge.setForeground(Color.BLACK);
            txtImageDetails.setText("Lỗi: " + (msg != null ? msg : jsonStr));
            return;
        }

        boolean isAlert = extractJsonBool(jsonStr, "is_alert");
        int weaponCount = extractJsonInt(jsonStr, "weapon_count");
        String outImg = extractJsonString(jsonStr, "output_image");

        if (outImg != null && new File(outImg).exists()) {
            displayScaledImage(lblProcessedImage, outImg);
        }

        StringBuilder details = new StringBuilder();
        if (isAlert && weaponCount > 0) {
            // 🚨 BẬT CẢNH BÁO ĐỎ RỰC
            lblImageAlertBadge.setText("🚨 CẢNH BÁO ĐỎ: PHÁT HIỆN " + weaponCount + " HUNG KHÍ NGUY HIỂM!");
            lblImageAlertBadge.setBackground(new Color(220, 20, 60)); // Crimson Red
            lblImageAlertBadge.setForeground(Color.WHITE);
            details.append("🚨 NGUY HIỂM - CẢNH BÁO ĐỎ ĐÃ ĐƯỢC KÍCH HOẠT!\n");
            details.append("• Đã khoanh vùng radar và gắn nhãn hung khí.\n");
            details.append("• Số hung khí xác nhận: ").append(weaponCount).append("\n");
        } else {
            lblImageAlertBadge.setText("🟢 AN TOÀN: Không phát hiện hung khí");
            lblImageAlertBadge.setBackground(new Color(40, 167, 69)); // Green
            lblImageAlertBadge.setForeground(Color.WHITE);
            details.append("🟢 AN TOÀN - Khung hình không chứa hung khí nguy hiểm.\n");
        }

        int rejectedCount = 0;
        java.util.regex.Matcher rm = java.util.regex.Pattern.compile("\"reason\"\\s*:").matcher(jsonStr);
        while (rm.find()) rejectedCount++;
        if (rejectedCount > 0) {
            details.append("• Vật thể mờ / dưới ngưỡng lọc (đã loại bỏ): ").append(rejectedCount).append("\n");
        }
        if (outImg != null) {
            details.append("• File ảnh kết quả: ").append(outImg).append("\n");
        }
        details.append("• Dữ liệu JSON: ").append(jsonStr);
        txtImageDetails.setText(details.toString());
    }

    // =========================================================================
    // 2. THẺ: XỬ LÝ VIDEO
    // =========================================================================
    private JPanel createVideoTab() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(new EmptyBorder(10, 10, 10, 10));

        JPanel toolBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 5));
        JButton btnSelectVideo = new JButton("📂 Chọn Video...");
        JButton btnScanVideo = new JButton("▶️ Bắt Đầu Quét Video");
        btnScanVideo.setEnabled(false);
        btnScanVideo.setBackground(new Color(0, 123, 255));
        btnScanVideo.setForeground(Color.WHITE);

        btnOpenResultVideo = new JButton("🎬 Mở Video Đã Khoanh Vùng");
        btnOpenResultVideo.setEnabled(false);

        lblVideoAlertBadge = new JLabel("Trạng thái: Chưa chọn video");
        lblVideoAlertBadge.setFont(new Font("Segoe UI", Font.BOLD, 13));
        lblVideoAlertBadge.setOpaque(true);
        lblVideoAlertBadge.setBackground(new Color(230, 230, 230));
        lblVideoAlertBadge.setBorder(new EmptyBorder(5, 12, 5, 12));

        toolBar.add(btnSelectVideo);
        toolBar.add(btnScanVideo);
        toolBar.add(btnOpenResultVideo);
        toolBar.add(lblVideoAlertBadge);
        panel.add(toolBar, BorderLayout.NORTH);

        JPanel centerPanel = new JPanel(new BorderLayout(10, 10));
        videoProgressBar = new JProgressBar();
        videoProgressBar.setStringPainted(true);
        videoProgressBar.setString("Sẵn sàng");
        centerPanel.add(videoProgressBar, BorderLayout.NORTH);

        txtVideoDetails = new JTextArea();
        txtVideoDetails.setEditable(false);
        txtVideoDetails.setFont(new Font("Consolas", Font.PLAIN, 12));
        txtVideoDetails.setText("Vui lòng chọn video (.mp4) để hệ thống quét từng khung hình bằng YOLO26s + ByteTrack...");
        centerPanel.add(new JScrollPane(txtVideoDetails), BorderLayout.CENTER);

        panel.add(centerPanel, BorderLayout.CENTER);

        btnSelectVideo.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            File defaultDir = new File(WORK_DIR, "data/raw_videos");
            if (!defaultDir.exists()) defaultDir = WORK_DIR;
            chooser.setCurrentDirectory(defaultDir);

            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                selectedVideoFile = chooser.getSelectedFile();
                btnScanVideo.setEnabled(true);
                lblVideoAlertBadge.setText("Sẵn sàng quét: " + selectedVideoFile.getName());
                lblVideoAlertBadge.setBackground(new Color(220, 240, 255));
                lblVideoAlertBadge.setForeground(Color.BLACK);
                txtVideoDetails.setText("Đã chọn video: " + selectedVideoFile.getAbsolutePath() + "\nBấm 'Bắt Đầu Quét Video' để thực hiện.");
            }
        });

        btnScanVideo.addActionListener(e -> {
            if (selectedVideoFile == null) return;
            btnScanVideo.setEnabled(false);
            btnOpenResultVideo.setEnabled(false);
            videoProgressBar.setIndeterminate(true);
            videoProgressBar.setString("Đang khởi động tiến trình quét video...");
            lblVideoAlertBadge.setText("⏳ Đang quét video từng khung hình...");
            lblVideoAlertBadge.setBackground(Color.YELLOW);
            lblVideoAlertBadge.setForeground(Color.BLACK);
            txtVideoDetails.setText("Đang xử lý video: " + selectedVideoFile.getName() + " bằng GPU RTX 4050...\n");

            SwingWorker<String, String> worker = new SwingWorker<>() {
                @Override
                protected String doInBackground() throws Exception {
                    ProcessBuilder pb = new ProcessBuilder(
                            getPythonExe(), getBridgeScript(),
                            "--action", "video",
                            "--input", selectedVideoFile.getAbsolutePath(),
                            "--imgsz", "960"
                    );
                    pb.directory(WORK_DIR);
                    pb.environment().put("PYTHONIOENCODING", "utf-8");
                    pb.environment().put("PYTHONUTF8", "1");
                    pb.redirectErrorStream(true);
                    Process process = pb.start();

                    StringBuilder fullLog = new StringBuilder();
                    String lastJson = null;

                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            fullLog.append(line).append("\n");
                            String trimmed = line.trim();
                            if (trimmed.contains("%") || trimmed.contains("FPS")) {
                                publish(trimmed);
                            }
                            if (trimmed.startsWith("{") && trimmed.contains("\"status\"")) {
                                lastJson = trimmed;
                            }
                        }
                    }
                    int exitCode = process.waitFor();
                    if (lastJson != null) {
                        return lastJson;
                    }
                    throw new RuntimeException("Lỗi xử lý video (Exit " + exitCode + "):\n" + fullLog);
                }

                @Override
                protected void process(java.util.List<String> chunks) {
                    if (!chunks.isEmpty()) {
                        String latest = chunks.get(chunks.size() - 1);
                        videoProgressBar.setIndeterminate(false);
                        videoProgressBar.setString(latest);
                        txtVideoDetails.setText("Tiến độ: " + latest);
                    }
                }

                @Override
                protected void done() {
                    btnScanVideo.setEnabled(true);
                    videoProgressBar.setIndeterminate(false);

                    try {
                        String jsonStr = get();
                        String status = extractJsonString(jsonStr, "status");
                        if ("error".equals(status)) {
                            String msg = extractJsonString(jsonStr, "message");
                            lblVideoAlertBadge.setText("❌ Lỗi: " + (msg != null ? msg : jsonStr));
                            lblVideoAlertBadge.setBackground(Color.PINK);
                            lblVideoAlertBadge.setForeground(Color.BLACK);
                            txtVideoDetails.setText("LỖI XỬ LÝ VIDEO:\n" + (msg != null ? msg : jsonStr));
                            videoProgressBar.setString("Lỗi");
                            return;
                        }

                        String outVid = extractJsonString(jsonStr, "output_video");
                        if (outVid != null) {
                            resultVideoFile = new File(outVid);
                            btnOpenResultVideo.setEnabled(true);
                            videoProgressBar.setValue(100);
                            videoProgressBar.setString("100% Hoàn tất!");

                            lblVideoAlertBadge.setText("🚨 HOÀN TẤT: ĐÃ KHOANH VÙNG & GẮN NHÃN VIDEO!");
                            lblVideoAlertBadge.setBackground(new Color(220, 20, 60));
                            lblVideoAlertBadge.setForeground(Color.WHITE);

                            txtVideoDetails.setText("✅ XỬ LÝ VIDEO THÀNH CÔNG!\n\n" +
                                    "• File video kết quả: " + outVid + "\n" +
                                    "• Bấm nút [🎬 Mở Video Đã Khoanh Vùng] để phát video ngay.\n" +
                                    "• Các khung hình có hung khí đã được đóng khung radar, dán nhãn tiếng Việt và viền đỏ báo động.");
                        } else {
                            lblVideoAlertBadge.setText("❌ Không tìm thấy đường dẫn video kết quả.");
                        }
                    } catch (Exception ex) {
                        lblVideoAlertBadge.setText("❌ Lỗi: " + ex.getMessage());
                        lblVideoAlertBadge.setBackground(Color.PINK);
                        lblVideoAlertBadge.setForeground(Color.BLACK);
                        txtVideoDetails.setText("LỖI XỬ LÝ VIDEO:\n" + ex.getMessage());
                        videoProgressBar.setString("Lỗi");
                    }
                }
            };
            worker.execute();
        });

        btnOpenResultVideo.addActionListener(e -> {
            if (resultVideoFile != null && resultVideoFile.exists()) {
                try {
                    Desktop.getDesktop().open(resultVideoFile);
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(this, "Không thể mở video: " + ex.getMessage());
                }
            }
        });

        return panel;
    }

    // =========================================================================
    // 3. THẺ: CAMERA TRỰC TIẾP
    // =========================================================================
    private JPanel createCameraTab() {
        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(new EmptyBorder(10, 10, 10, 10));

        // Toolbar
        JPanel toolBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 5));
        JButton btnStartCam = new JButton("🔴 Bật Camera Giám Sát");
        btnStartCam.setFont(new Font("Segoe UI", Font.BOLD, 12));
        btnStartCam.setBackground(new Color(220, 53, 69));
        btnStartCam.setForeground(Color.WHITE);

        JButton btnStopCam = new JButton("⏹️ Dừng Camera");
        btnStopCam.setFont(new Font("Segoe UI", Font.BOLD, 12));
        btnStopCam.setEnabled(false);

        comboCameraSource = new JComboBox<>(new String[]{
                "📷 Webcam Mặc Định (Index 0)",
                "📷 Webcam Phụ (Index 1)",
                "📹 Mô Phỏng CCTV (Video Mẫu: test_video_2.mp4)"
        });
        comboCameraSource.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        lblCameraAlertBadge = new JLabel("Trạng thái: Camera đang tắt");
        lblCameraAlertBadge.setFont(new Font("Segoe UI", Font.BOLD, 13));
        lblCameraAlertBadge.setOpaque(true);
        lblCameraAlertBadge.setBackground(new Color(230, 230, 230));
        lblCameraAlertBadge.setBorder(new EmptyBorder(5, 12, 5, 12));

        toolBar.add(btnStartCam);
        toolBar.add(btnStopCam);
        toolBar.add(new JLabel(" Nguồn:"));
        toolBar.add(comboCameraSource);
        toolBar.add(lblCameraAlertBadge);
        panel.add(toolBar, BorderLayout.NORTH);

        // Màn hình Camera hiển thị trực tiếp trong Swing Panel
        lblCameraScreen = new JLabel("<html><center style='color:#a0b0c8;'><br><br><span style='font-size:22px;'>🎥 MÀN HÌNH CAMERA ĐANG TẮT</span><br><br><span style='font-size:13px;color:#8090a8;'>Nhấn nút <b>[🔴 Bật Camera Giám Sát]</b> phía trên để bắt đầu luồng nhận diện thời gian thực.</span><br><span style='font-size:12px;color:#607088;'>Hệ thống sẽ tự động khoanh vùng radar, gắn nhãn tiếng Việt và bật viền cảnh báo đỏ khi phát hiện hung khí.</span></center></html>", SwingConstants.CENTER);
        lblCameraScreen.setBackground(new Color(15, 18, 26));
        lblCameraScreen.setOpaque(true);
        lblCameraScreen.setBorder(BorderFactory.createLineBorder(new Color(50, 60, 80), 3));

        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.setBorder(new TitledBorder(new LineBorder(new Color(60, 75, 100)), "Màn Hình Giám Sát Thời Gian Thực (Live Camera / CCTV View)", TitledBorder.LEFT, TitledBorder.TOP, new Font("Segoe UI", Font.BOLD, 12), Color.DARK_GRAY));
        centerPanel.add(lblCameraScreen, BorderLayout.CENTER);

        // Thanh thông tin bên dưới màn hình
        lblCameraInfo = new JLabel(" Hệ thống sẵn sàng. Chọn nguồn và bấm Bật Camera.", SwingConstants.LEFT);
        lblCameraInfo.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        lblCameraInfo.setOpaque(true);
        lblCameraInfo.setBackground(new Color(240, 243, 248));
        lblCameraInfo.setBorder(new EmptyBorder(6, 10, 6, 10));
        centerPanel.add(lblCameraInfo, BorderLayout.SOUTH);

        panel.add(centerPanel, BorderLayout.CENTER);

        // Xử lý sự kiện Bật Camera
        btnStartCam.addActionListener(e -> {
            btnStartCam.setEnabled(false);
            btnStopCam.setEnabled(true);
            comboCameraSource.setEnabled(false);
            isCameraRunning = true;

            lblCameraAlertBadge.setText("🟡 ĐANG KẾT NỐI CAMERA & NẠP MODEL...");
            lblCameraAlertBadge.setBackground(new Color(255, 193, 7));
            lblCameraAlertBadge.setForeground(Color.BLACK);
            lblCameraScreen.setText("<html><center style='color:#f0ad4e;'><span style='font-size:18px;'>Đang khởi động Camera và nạp mô hình YOLO26s...</span><br>Vui lòng đợi trong giây lát.</center></html>");

            int sourceIdx = comboCameraSource.getSelectedIndex();

            new Thread(() -> {
                try {
                    java.util.List<String> command = new java.util.ArrayList<>();
                    command.add(getPythonExe());
                    command.add(getBridgeScript());
                    command.add("--action");
                    command.add("camera");

                    if (sourceIdx == 0) {
                        command.add("--cam");
                        command.add("0");
                    } else if (sourceIdx == 1) {
                        command.add("--cam");
                        command.add("1");
                    } else {
                        command.add("--input");
                        command.add("data/raw_videos/test_video_2.mp4");
                    }

                    ProcessBuilder pb = new ProcessBuilder(command);
                    pb.directory(new File(".").getAbsoluteFile());
                    pb.environment().put("PYTHONIOENCODING", "utf-8");
                    pb.environment().put("PYTHONUTF8", "1");
                    pb.redirectError(ProcessBuilder.Redirect.INHERIT); // Để warning/log đi ra console mà không hỏng stream
                    cameraProcess = pb.start();

                    DataInputStream dis = new DataInputStream(new BufferedInputStream(cameraProcess.getInputStream()));

                    while (isCameraRunning && cameraProcess.isAlive()) {
                        int metaLen;
                        try {
                            metaLen = dis.readInt();
                        } catch (Exception ioEx) {
                            break;
                        }

                        byte[] metaBytes = new byte[metaLen];
                        dis.readFully(metaBytes);
                        String metaJson = new String(metaBytes, StandardCharsets.UTF_8);

                        int imgLen = dis.readInt();
                        if (imgLen <= 0) {
                            SwingUtilities.invokeLater(() -> {
                                lblCameraScreen.setText("<html><center style='color:#ff6b6b;font-size:14px;padding:20px;'>" + metaJson + "</center></html>");
                            });
                            break;
                        }

                        byte[] imgBytes = new byte[imgLen];
                        dis.readFully(imgBytes);

                        BufferedImage frameImg = ImageIO.read(new ByteArrayInputStream(imgBytes));
                        if (frameImg != null) {
                            SwingUtilities.invokeLater(() -> {
                                if (!isCameraRunning) return;

                                int pw = lblCameraScreen.getWidth();
                                int ph = lblCameraScreen.getHeight();
                                if (pw > 100 && ph > 100) {
                                    double scale = Math.min((double) pw / frameImg.getWidth(), (double) ph / frameImg.getHeight());
                                    int nw = Math.max(1, (int) (frameImg.getWidth() * scale));
                                    int nh = Math.max(1, (int) (frameImg.getHeight() * scale));
                                    Image scaled = frameImg.getScaledInstance(nw, nh, Image.SCALE_FAST);
                                    lblCameraScreen.setIcon(new ImageIcon(scaled));
                                } else {
                                    lblCameraScreen.setIcon(new ImageIcon(frameImg));
                                }
                                lblCameraScreen.setText("");

                                boolean isAlert = metaJson.contains("\"is_alert\": true");
                                if (isAlert) {
                                    lblCameraScreen.setBorder(BorderFactory.createLineBorder(new Color(220, 20, 60), 4));
                                    lblCameraAlertBadge.setText("🚨 CẢNH BÁO ĐỎ: PHÁT HIỆN HUNG KHÍ!");
                                    lblCameraAlertBadge.setBackground(new Color(220, 53, 69));
                                    lblCameraAlertBadge.setForeground(Color.WHITE);
                                } else {
                                    lblCameraScreen.setBorder(BorderFactory.createLineBorder(new Color(40, 167, 69), 3));
                                    lblCameraAlertBadge.setText("🟢 AN TOÀN - ĐANG QUÉT...");
                                    lblCameraAlertBadge.setBackground(new Color(40, 167, 69));
                                    lblCameraAlertBadge.setForeground(Color.WHITE);
                                }

                                lblCameraInfo.setText("  " + metaJson.replace("{", "").replace("}", ""));
                            });
                        }
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                } finally {
                    isCameraRunning = false;
                    if (cameraProcess != null && cameraProcess.isAlive()) {
                        cameraProcess.destroyForcibly();
                    }
                    SwingUtilities.invokeLater(() -> {
                        btnStartCam.setEnabled(true);
                        btnStopCam.setEnabled(false);
                        comboCameraSource.setEnabled(true);
                        lblCameraScreen.setIcon(null);
                        lblCameraScreen.setText("<html><center style='color:#a0b0c8;'><br><span style='font-size:20px;'>MÀN HÌNH CAMERA ĐÃ DỪNG</span><br><br><span style='font-size:13px;color:#8090a8;'>Nhấn [🔴 Bật Camera Giám Sát] để mở lại luồng.</span></center></html>");
                        lblCameraScreen.setBorder(BorderFactory.createLineBorder(new Color(50, 60, 80), 3));
                        lblCameraAlertBadge.setText("Trạng thái: Camera đang tắt");
                        lblCameraAlertBadge.setBackground(new Color(230, 230, 230));
                        lblCameraAlertBadge.setForeground(Color.BLACK);
                        lblCameraInfo.setText(" Camera đã dừng kết nối.");
                    });
                }
            }).start();
        });

        // Xử lý sự kiện Dừng Camera
        btnStopCam.addActionListener(e -> {
            isCameraRunning = false;
            if (cameraProcess != null && cameraProcess.isAlive()) {
                cameraProcess.destroyForcibly();
            }
        });

        return panel;
    }

    // Tiện ích hiển thị ảnh co giãn theo khung
    private void displayScaledImage(JLabel label, String imagePath) {
        ImageIcon icon = new ImageIcon(imagePath);
        Image img = icon.getImage();
        int w = label.getWidth() > 0 ? label.getWidth() : 450;
        int h = label.getHeight() > 0 ? label.getHeight() : 350;

        double scale = Math.min((double) w / img.getWidth(null), (double) h / img.getHeight(null));
        int nw = Math.max(1, (int) (img.getWidth(null) * scale));
        int nh = Math.max(1, (int) (img.getHeight(null) * scale));

        Image scaled = img.getScaledInstance(nw, nh, Image.SCALE_SMOOTH);
        label.setIcon(new ImageIcon(scaled));
        label.setText("");
    }

    // =========================================================================
    // HÀM MAIN: Cho phép chạy độc lập hoặc nhúng làm một tab trong App chính
    // =========================================================================
    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}

        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Hệ Thống Trinh Sát Phát Hiện Hung Khí (Java Swing)");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(1050, 720);
            frame.setLocationRelativeTo(null);

            // Nhúng thẻ phát hiện hung khí vào Frame
            frame.add(new WeaponDetectionPanel());
            frame.setVisible(true);
        });
    }
}

