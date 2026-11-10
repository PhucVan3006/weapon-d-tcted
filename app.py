"""
app.py - GIAO DIỆN WEB MỘT FILE cho hệ thống phát hiện hung khí.

Ba chức năng trong một trang:
    1. Camera trực tiếp   - quét webcam theo thời gian thực
    2. Quét video         - tải video lên, xuất video đã khoanh vùng
    3. Quét ảnh           - tải một hoặc nhiều ảnh

Toàn bộ dùng CHUNG lõi src/common/detector.py, nên nhãn, ngưỡng, màu sắc
và luật phân mức ở cả ba tab luôn giống hệt nhau. Đây chính là điều mà
kiến trúc cũ (bốn file tự xử lý riêng) không bảo đảm được.

CÀI ĐẶT
-------
    pip install gradio

CHẠY
----
    python app.py
    python app.py --share          # tạo link public tạm thời
    python app.py --port 8080

Mở trình duyệt tại http://127.0.0.1:7860
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import warnings
warnings.filterwarnings("ignore")

from src.common.detector import WeaponDetector, safe_crop

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT_DIR = Path("storage/web_results")
DETECTOR = None          # nạp một lần, dùng lại cho cả ba tab
AGENT = None


def _draw_downgraded_banner(img, v):
    """
    Vẽ lại dải trạng thái khi agent hạ mức cảnh báo.

    Nếu không vẽ lại thì HUD vẫn hiện "CANH BAO: n VU KHI" màu đỏ trong khi
    phần chữ bên dưới nói đã hạ mức - hai nơi nói hai điều trái nhau.
    """
    h, w = img.shape[:2]
    band = img.copy()
    cv2.rectangle(band, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.addWeighted(band, 0.72, img, 0.28, 0, img)
    cv2.putText(img, "DA HA MUC - AGENT: KHONG PHAI MOI DE DOA", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 0), 2,
                lineType=cv2.LINE_AA)


def get_agent():
    global AGENT
    if AGENT is None:
        try:
            from src.agent.security_agent import SecurityAnalystAgent
            AGENT = SecurityAnalystAgent()
        except Exception as e:
            print(f"[!] Không thể nạp SecurityAnalystAgent: {e}")
            AGENT = None
    return AGENT


def get_detector(imgsz, conf_mode, conf_value, weapons_only, use_geometry, tta):
    """
    Tạo (hoặc tạo lại) lõi phát hiện khi người dùng đổi tuỳ chọn.
    Mô hình chỉ được nạp lại khi thật sự cần.
    """
    global DETECTOR
    override = None if conf_mode == "Ngưỡng riêng từng lớp" else float(conf_value)
    key = (int(imgsz), override, bool(weapons_only), bool(use_geometry), bool(tta))

    if DETECTOR is None or getattr(DETECTOR, "_key", None) != key:
        DETECTOR = WeaponDetector(
            imgsz=int(imgsz), conf_override=override,
            weapons_only=bool(weapons_only), use_geometry=bool(use_geometry),
            augment=bool(tta),
        )
        DETECTOR._key = key
        print("\n" + DETECTOR.describe() + "\n")
    return DETECTOR


def _settings_note(det, rejected):
    note = ""
    if rejected:
        from collections import Counter
        why = Counter(r[2] for r in rejected)
        note = "\n\nĐã loại " + str(len(rejected)) + " ứng viên:\n"
        note += "\n".join(f"   - {k}: {v}" for k, v in why.most_common(4))
    return note


# ---------------------------------------------------------------------------
# TAB 1 & 3 - ảnh tĩnh và khung camera
# ---------------------------------------------------------------------------
def scan_image(image, imgsz, conf_mode, conf_value, weapons_only, use_geometry, tta, use_agent=True):
    if image is None:
        return None, "Chưa có ảnh nào."
    det = get_detector(imgsz, conf_mode, conf_value, weapons_only, use_geometry, tta)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    dets, rejected = det.detect(frame)
    ms = (time.perf_counter() - t0) * 1000
    canvas = det.draw(frame, dets)

    txt = det.summarize(dets)
    txt += f"\n\nThời gian xử lý YOLO26s: {ms:.0f} ms"
    txt += _settings_note(det, rejected)

    if dets and use_agent:
        try:
            agent = get_agent()
            if agent and agent.is_available():
                primary = max(dets, key=lambda d: d.conf)
                # safe_crop KẸP BIÊN. Viết thẳng frame[y1:y2, x1:x2] với toạ
                # độ hơi âm (YOLO vẫn trả ra) thì numpy không báo lỗi mà cắt
                # một vùng SAI - ảnh gửi cho agent là chỗ khác hẳn vật thể.
                crop = safe_crop(frame, primary.box, pad=8)
                rep = agent.analyze_threat(
                    frame, crop=crop if crop.size else None,
                    class_name=primary.cls_name, confidence=primary.conf)
                v = agent.verdict(rep)

                if v["ok"]:
                    badge = {
                        1: "🟢 AN TOÀN / BÌNH THƯỜNG",
                        2: "🟡 NGHI VẤN / CẦN QUAN SÁT",
                        3: "🔴 NGUY HIỂM KHẨN CẤP",
                    }[v["threat_level"]]
                    txt += f"\n\n{'='*45}\n🤖 THẨM ĐỊNH BỐI CẢNH (GEMINI AI AGENT)\n{'='*45}\n"
                    txt += f"• Vật được thẩm định: {primary.vi_name} ({primary.conf*100:.0f}%)\n"
                    txt += f"• Mức độ đe dọa: {badge}\n"
                    txt += f"• Bối cảnh: {v['context'] or 'N/A'}\n"
                    txt += f"• Hành vi đối tượng: {v['person_behavior'] or 'N/A'}\n"
                    txt += f"• Tóm tắt: {v['summary_vi'] or 'N/A'}\n"
                    txt += f"• Khuyến nghị: {v['action_recommended'] or 'N/A'}\n"
                    txt += f"• Độ trễ: {v['latency_ms']} ms | Model: {v['model']}"

                    # KẾT LUẬN CỦA AGENT PHẢI CÓ TÁC DỤNG, không chỉ in ra.
                    # Nhưng chỉ HẠ MỨC CẢNH BÁO - hộp vẫn vẽ, phát hiện vẫn
                    # còn. Mô hình ngôn ngữ cũng sai, và nếu nó nói sai cho
                    # một vũ khí thật thì xoá phát hiện là che mất mối nguy.
                    if v["downgrade_alert"]:
                        txt += ("\n\n⚠️  CẢNH BÁO ĐÃ ĐƯỢC HẠ MỨC: agent xác định "
                                "đây không phải mối đe dọa (sinh hoạt / lao động "
                                "bình thường).\n     Hộp vẫn được vẽ để bạn tự "
                                "kiểm tra — agent chỉ tắt còi, không xoá phát hiện.")
                        canvas = det.draw(frame, dets, show_hud=False)
                        _draw_downgraded_banner(canvas, v)
                else:
                    txt += (f"\n\n[AI Agent không thẩm định được: "
                            f"{rep.get('error', 'không rõ nguyên nhân')}]"
                            f"\n Cảnh báo giữ nguyên mức của YOLO.")
        except Exception as e:
            txt += f"\n\n[Lỗi AI Agent: {e}]"

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), txt


def scan_camera(frame, imgsz, conf_mode, conf_value, weapons_only, use_geometry):
    """Luồng webcam - TTA luôn tắt để giữ tốc độ."""
    if frame is None:
        return None, ""
    det = get_detector(imgsz, conf_mode, conf_value, weapons_only, use_geometry, False)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    dets, _ = det.detect(bgr)
    canvas = det.draw(bgr, dets)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), det.summarize(dets)


# ---------------------------------------------------------------------------
# TAB 2 - video
# ---------------------------------------------------------------------------
def scan_video(video_path, imgsz, conf_mode, conf_value, weapons_only,
               use_geometry, frame_stride, progress=None):
    if not video_path:
        return None, "Chưa chọn video."

    det = get_detector(imgsz, conf_mode, conf_value, weapons_only, use_geometry, False)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Không mở được video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, int(frame_stride))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(OUT_DIR / f"{Path(video_path).stem}_detected.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    idx = 0
    scanned = 0
    last_dets = []
    per_class = {}
    frames_with_weapon = 0
    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        # Chỉ chạy mô hình mỗi `stride` khung; các khung giữa dùng lại kết
        # quả gần nhất. stride=1 nghĩa là quét 100% số khung.
        if idx % stride == 0 or not last_dets:
            last_dets, _ = det.detect(frame)
            scanned += 1
            for d in last_dets:
                per_class[d.vi_name] = per_class.get(d.vi_name, 0) + 1
            if any(d.is_weapon for d in last_dets):
                frames_with_weapon += 1

        writer.write(det.draw(frame, last_dets, copy=False))

        if progress is not None and total and idx % 30 == 0:
            progress(idx / total, desc=f"Khung {idx}/{total}")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0

    lines = [
        f"Đã xử lý {idx} khung ({idx/fps:.1f} giây video) trong {elapsed:.1f} giây",
        f"Chạy mô hình trên {scanned} khung  (bước nhảy = {stride})",
        f"Tốc độ: {idx/elapsed:.1f} khung/giây",
        f"Số khung có vũ khí: {frames_with_weapon}",
        "",
    ]
    if per_class:
        lines.append("Số lần xuất hiện theo lớp:")
        for k, v in sorted(per_class.items(), key=lambda x: -x[1]):
            lines.append(f"   • {k}: {v}")
    else:
        lines.append("Không phát hiện vũ khí hay vật nghi vấn nào.")
    lines.append(f"\nVideo kết quả: {out_path}")
    return out_path, "\n".join(lines)


# ---------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    css = """
    .gradio-container {max-width: 1280px !important}
    footer {display: none !important}
    """

    with gr.Blocks(title="Hệ thống trinh sát phát hiện hung khí", css=css,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Hệ thống trinh sát phát hiện hung khí\n"
            "YOLO26s · Học viện Công nghệ Bưu chính Viễn thông (PTIT)"
        )

        # --- Bảng tuỳ chọn dùng chung cho cả ba tab --------------------
        with gr.Accordion("Tuỳ chọn phát hiện", open=False):
            with gr.Row():
                imgsz = gr.Dropdown([640, 960, 1280], value=960, label="Kích thước ảnh",
                                    info="Nên để KHỚP với imgsz lúc train (960). "
                                         "Lệch kích thước là nguyên nhân sót vật nhỏ rất thường gặp.")
                conf_mode = gr.Radio(
                    ["Ngưỡng riêng từng lớp", "Ép một ngưỡng chung"],
                    value="Ngưỡng riêng từng lớp", label="Cách đặt ngưỡng",
                    info="Ngưỡng riêng: súng 0.30 ... xích 0.60",
                )
                conf_value = gr.Slider(0.05, 0.9, value=0.35, step=0.05,
                                       label="Ngưỡng chung (khi chọn ép)")
            with gr.Row():
                weapons_only = gr.Checkbox(
                    value=False, label="Chỉ vũ khí thật",
                    info="Bỏ qua nhóm dụng cụ: Crowbar, Stick, Hammer, Spanner, Chain")
                use_geometry = gr.Checkbox(
                    value=True, label="Lọc hình học",
                    info="Loại hộp quá to — thường là cánh tay, balo, mép tường")
                tta = gr.Checkbox(
                    value=False, label="TTA (ảnh tĩnh)",
                    info="Chính xác hơn 1-2 điểm mAP, chậm hơn khoảng 3 lần")

        opts = [imgsz, conf_mode, conf_value, weapons_only, use_geometry]

        with gr.Tabs():
            # ---------------- TAB 1: CAMERA ----------------
            with gr.Tab("Camera trực tiếp"):
                gr.Markdown(
                    "Cho phép trình duyệt dùng webcam, rồi bấm **Bật camera**. "
                    "Mỗi khung hình được quét ngay khi nhận được."
                )
                with gr.Row():
                    cam_in = gr.Image(sources=["webcam"], streaming=True,
                                      label="Nguồn camera", height=420)
                    cam_out = gr.Image(label="Kết quả phát hiện", height=420)
                cam_txt = gr.Textbox(label="Tình trạng", lines=5)
                cam_in.stream(fn=scan_camera, inputs=[cam_in] + opts,
                              outputs=[cam_out, cam_txt],
                              stream_every=0.2, show_progress="hidden")

            # ---------------- TAB 2: VIDEO ----------------
            with gr.Tab("Quét video"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Video đầu vào")
                        stride = gr.Slider(
                            1, 10, value=2, step=1, label="Bước nhảy khung",
                            info="1 = quét mọi khung (chính xác nhất). "
                                 "2 = quét một nửa, nhanh gấp đôi")
                        vid_btn = gr.Button("Bắt đầu quét", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="Video đã khoanh vùng")
                        vid_txt = gr.Textbox(label="Báo cáo", lines=14)

                def _run_video(v, i, cm, cv_, wo, ug, st,
                               progress=gr.Progress()):
                    return scan_video(v, i, cm, cv_, wo, ug, st, progress)

                vid_btn.click(fn=_run_video, inputs=[vid_in] + opts + [stride],
                              outputs=[vid_out, vid_txt])

            # ---------------- TAB 3: ẢNH ----------------
            with gr.Tab("Quét ảnh"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(label="Ảnh đầu vào", type="numpy", height=420)
                        use_agent = gr.Checkbox(
                            value=True,
                            label="🤖 Kích hoạt AI Agent (Gemini) thẩm định bối cảnh & hành vi",
                            info="Tự động phân biệt hoạt động lao động/nấu ăn bình thường vs hành vi đe dọa vũ lực"
                        )
                        img_btn = gr.Button("Quét ảnh & Phân tích AI Agent", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(label="Kết quả", height=420)
                        img_txt = gr.Textbox(label="Chi tiết phát hiện & Báo cáo AI Agent", lines=16)
                img_btn.click(fn=scan_image, inputs=[img_in] + opts + [tta, use_agent],
                              outputs=[img_out, img_txt])

        gr.Markdown(
            "---\n"
            "**Đỏ đậm** = vũ khí thật (báo động) · **vàng nét mảnh `[NV]`** = "
            "vật nghi vấn, là dụng cụ chứ không phải vũ khí.\n\n"
            "Ngưỡng từng lớp và luật phân mức đặt trong "
            "`configs/system_config.yaml`, mục `class_policy`."
        )
    return demo


def main():
    ap = argparse.ArgumentParser(description="Giao diện web phát hiện hung khí")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Tạo link public tạm thời")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        print("=" * 70)
        print("  Chưa cài gradio. Chạy lệnh sau rồi mở lại:")
        print("      pip install gradio")
        print("=" * 70)
        return

    print("=" * 70)
    print("  HỆ THỐNG TRINH SÁT PHÁT HIỆN HUNG KHÍ - GIAO DIỆN WEB")
    print(f"  Mở trình duyệt tại: http://{args.host}:{args.port}")
    print("=" * 70)

    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
