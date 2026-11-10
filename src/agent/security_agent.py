"""
Security Analyst AI Agent (Gemini Powered).
Phân tích ngữ cảnh an ninh chuyên sâu kết hợp với mô hình YOLO26s.

Nhiệm vụ:
1. Đọc bối cảnh không gian (nhà bếp, công trường, quầy thu ngân, cây ATM, đường phố...).
2. Phân tích tư thế & ý định của đối tượng cầm vật thể (đe dọa, tấn công vs lao động, sinh hoạt).
3. Đánh giá mức độ đe dọa (1: An toàn, 2: Cần giám sát, 3: Nguy hiểm khẩn cấp).
4. Đưa ra khuyến nghị xử lý và soạn tin nhắn cảnh báo gửi lực lượng an ninh.
"""

import os
import io
import sys
import json
import time
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Nạp biến môi trường từ .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class SecurityAnalystAgent:
    """
    Nhân viên phân tích an ninh AI sử dụng Google Gemini.
    Được kích hoạt khi YOLO26s phát hiện hung khí được xác thực qua ByteTrack.
    """

    DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # BỎ DANH SÁCH MODEL GÁN CỨNG.
    # Bản cũ: FALLBACK_MODELS = [..., "gemini-3.8-flash"] - một tên không
    # kiểm chứng được, và danh sách tên model của Google thay đổi theo thời
    # gian nên mọi danh sách gán cứng đều sẽ mục. Nay hỏi thẳng API xem có
    # model nào (`client.models.list()`), chỉ dùng tên lấy được từ đó.
    # Ưu tiên model nhẹ/nhanh vì mỗi khung hình là một lần gọi.
    PREFER_HINTS = ("flash-lite", "flash")

    # Thời gian chờ mỗi lần gọi. Không có nó thì một lần gọi treo sẽ chặn
    # luôn luồng cảnh báo (Thread-3) vô thời hạn.
    TIMEOUT_MS = 20_000

    def __init__(self, api_key=None, model_name=None, quiet=False):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.client = None
        self._listed = None          # cache danh sách model thật, hỏi 1 lần
        self._working_model = None   # model đã gọi thành công, dùng lại
        self._fatal = None           # lỗi không thể khắc phục (sai key, hết quota)

        if not self.api_key:
            if not quiet:
                print("[CẢNH BÁO] Không tìm thấy GEMINI_API_KEY. Thiết lập trong file .env.")
            return

        if genai is None:
            if not quiet:
                print("[CẢNH BÁO] Chưa cài google-genai. Chạy: pip install google-genai")
            return

        try:
            # http_options có thể khác giữa các bản google-genai, nên thử
            # kèm timeout trước, thất bại thì tạo client thường.
            try:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=self.TIMEOUT_MS),
                )
            except Exception:
                self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[!] Lỗi khởi tạo Gemini Client: {e}")
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None and self._fatal is None

    # ------------------------------------------------------------------
    @staticmethod
    def _is_fatal(err) -> bool:
        """
        Lỗi KHÔNG nên thử model khác: sai khoá, không có quyền, hết quota.

        Bản cũ bắt mọi Exception rồi `continue` sang model tiếp theo, nên một
        khoá API sai phải chờ hết 4 lần gọi thất bại mới trả về - và lần
        chạy sau lại chờ tiếp từ đầu.
        """
        s = f"{type(err).__name__}: {err}".lower()
        for kw in ("api key", "api_key", "unauthenticated", "unauthorized",
                   "permission", "forbidden", "quota", "resource_exhausted",
                   "billing", "401", "403", "429"):
            if kw in s:
                return True
        return False

    def available_models(self):
        """
        Tên model THẬT mà khoá API này gọi được, hỏi trực tiếp từ API.
        Hỏi một lần rồi nhớ. Trả về [] nếu không hỏi được.
        """
        if self._listed is not None:
            return self._listed
        names = []
        try:
            for m in self.client.models.list():
                name = getattr(m, "name", "") or ""
                short = name.split("/")[-1]
                actions = getattr(m, "supported_actions", None) or \
                    getattr(m, "supported_generation_methods", None) or []
                # Một số bản SDK không trả trường này -> không loại bỏ gì.
                if actions and not any("generatecontent" in str(a).lower().replace("_", "")
                                       for a in actions):
                    continue
                if short:
                    names.append(short)
        except Exception as e:
            print(f"[!] Không liệt kê được model Gemini: {e}")
            self._listed = []
            return self._listed

        def rank(n):
            low = n.lower()
            for i, h in enumerate(self.PREFER_HINTS):
                if h in low:
                    return i
            return len(self.PREFER_HINTS)

        self._listed = sorted(names, key=lambda n: (rank(n), n))
        return self._listed

    def _candidates(self):
        """Model đã chạy được trước đó, rồi model cấu hình, rồi từ API."""
        out = []
        for c in (self._working_model, self.model_name):
            if c and c not in out:
                out.append(c)
        for c in self.available_models():
            if c not in out:
                out.append(c)
        return out

    def analyze_threat(
        self,
        full_frame: np.ndarray,
        crop: np.ndarray = None,
        class_name: str = "Weapon",
        confidence: float = 0.5,
        track_id: int = 0
    ) -> dict:
        """
        Gửi ảnh toàn cảnh và ảnh cắt hung khí cho Gemini phân tích ngữ cảnh.
        Trả về kết quả đánh giá có cấu trúc (dict).
        """
        if not self.is_available():
            return {
                "status": "unavailable",
                "threat_level": 2,
                "summary_vi": f"Phát hiện {class_name} (conf={confidence:.2f}), AI Agent chưa được cấu hình.",
                "action_recommended": "Kiểm tra thủ công qua camera."
            }

        t0 = time.perf_counter()

        # Chuẩn bị ảnh PIL từ BGR OpenCV frame
        rgb_full = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        pil_full = Image.fromarray(rgb_full)

        contents = [pil_full]

        if crop is not None and crop.size > 0:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(rgb_crop)
            contents.append("Dưới đây là ảnh chụp cận cảnh vật thể được phát hiện:")
            contents.append(pil_crop)

        prompt = f"""
Bạn là Chuyên gia phân tích an ninh giám sát (CCTV Security Analyst).
Hệ thống thị giác máy tính YOLO26s vừa phát hiện một vật thể nghi vấn trong khung hình:
- Loại vật thể: {class_name}
- Độ tin cậy phát hiện: {confidence * 100:.1f}%
- Mã định danh theo dõi: #{track_id}

Hãy quan sát kỹ toàn bộ khung cảnh và trả lời chính xác bằng tiếng Việt theo định dạng JSON với các trường sau:
{{
  "context": "<Mô tả không gian, địa điểm, thời điểm (ví dụ: quầy thu ngân, nhà bếp, đường phố, cây ATM, công trường)>",
  "person_behavior": "<Mô tả tư thế, cử chỉ và hành vi của người đang cầm/ở gần vật thể: có đe dọa, chĩa hung khí, che giấu mặt mũi hay chỉ đang làm việc/sinh hoạt bình thường?>",
  "threat_level": <Số nguyên: 1 (An toàn - sinh hoạt/nấu ăn/công việc bình thường), 2 (Nghi vấn - cần bảo vệ chú ý quan sát), 3 (Nguy hiểm khẩn cấp - đe dọa bạo lực, cướp, tấn công)>,
  "is_false_alarm": <true nếu là nhầm lẫn/vật dụng bình thường hoặc người đang lao động hợp pháp, false nếu là mối đe dọa an ninh thật sự>,
  "summary_vi": "<Tóm tắt ngắn gọn tình huống trong 2 câu để hiển thị cho nhân viên an ninh>",
  "action_recommended": "<Khuyến nghị xử lý cụ thể, ví dụ: 'Bỏ qua - hoạt động nấu ăn', 'Điều động bảo vệ kiểm tra ngay lập tức', 'Kích hoạt báo động khẩn cấp'>"
}}

LƯU Ý: Chỉ trả về duy nhất chuỗi JSON hợp lệ, không kèm văn bản giải thích mở đầu hay kết thúc.
"""
        contents.append(prompt)

        last_error = None

        for model_name in self._candidates():
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                elapsed = time.perf_counter() - t0

                raw_text = (response.text or "").strip()
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()

                result = json.loads(raw_text)
                if not isinstance(result, dict):
                    raise ValueError("Gemini không trả về đối tượng JSON")
                result["status"] = "success"
                result["latency_ms"] = round(elapsed * 1000, 1)
                result["model"] = model_name
                self._working_model = model_name   # lần sau dùng luôn
                return result
            except Exception as e:
                last_error = e
                if self._is_fatal(e):
                    # Sai khoá / hết quota: thử model khác cũng vô ích, và
                    # ghi nhớ để các lần gọi sau không chờ lại từ đầu.
                    self._fatal = str(e)
                    print(f"[!] Gemini từ chối, không phải lỗi model: {e}")
                    break
                continue

        return {
            "status": "error",
            "error": str(last_error),
            "threat_level": 2,
            "summary_vi": f"Phát hiện {class_name} ({confidence*100:.0f}%). "
                          f"Không phân tích được bối cảnh.",
            "action_recommended": "Kiểm tra trực tiếp tại hiện trường."
        }

    # ------------------------------------------------------------------
    @staticmethod
    def verdict(report: dict) -> dict:
        """
        Chuẩn hoá kết luận của agent thành thứ hệ thống DÙNG ĐƯỢC.

        VÌ SAO CẦN: bản cũ lấy `threat_level` và `is_false_alarm` về rồi chỉ
        IN RA. Agent nói "an toàn, đây là người đang nấu ăn" mà hệ thống vẫn
        báo động đỏ y như cũ. Tức là tầng agent chỉ để trang trí, không chữa
        được lỗi báo nhầm - đúng thứ nó được thêm vào để chữa.

        Nguyên tắc an toàn: agent CHỈ ĐƯỢC HẠ MỨC CẢNH BÁO, KHÔNG được xoá
        phát hiện. Một mô hình ngôn ngữ cũng sai, và nếu nó nói sai cho một
        vũ khí thật thì xoá phát hiện đi là che mất mối nguy. Nên hộp vẫn
        được vẽ, bằng chứng vẫn được lưu; chỉ tiếng còi bị tắt.
        """
        ok = isinstance(report, dict) and report.get("status") == "success"
        lvl = report.get("threat_level", 2) if ok else 2
        try:
            lvl = int(lvl)
        except (TypeError, ValueError):
            lvl = 2
        lvl = min(3, max(1, lvl))
        false_alarm = bool(report.get("is_false_alarm", False)) if ok else False

        # Chỉ hạ còi khi agent trả lời THÀNH CÔNG và nói rõ là mức 1 hoặc
        # là báo nhầm. Agent lỗi / chưa cấu hình -> giữ nguyên cảnh báo.
        downgrade = ok and (lvl == 1 or false_alarm)
        return {
            "ok": ok,
            "threat_level": lvl,
            "is_false_alarm": false_alarm,
            "downgrade_alert": downgrade,
            "badge": {1: "AN TOAN", 2: "NGHI VAN", 3: "NGUY HIEM"}.get(lvl, "NGHI VAN"),
            "summary_vi": report.get("summary_vi", ""),
            "context": report.get("context", ""),
            "person_behavior": report.get("person_behavior", ""),
            "action_recommended": report.get("action_recommended", ""),
            "latency_ms": report.get("latency_ms", 0),
            "model": report.get("model", ""),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chạy thử nghiệm AI Agent Gemini")
    parser.add_argument("--image", "-i", default=None, help="Đường dẫn ảnh cần phân tích")
    parser.add_argument("--weapon", "-w", default="Knife", help="Tên hung khí phát hiện")
    args = parser.parse_args()

    agent = SecurityAnalystAgent()
    if not agent.is_available():
        print("[!] Không khởi tạo được Gemini Agent.")
        sys.exit(1)

    print(f"[+] Khởi tạo AI Agent thành công với mô hình: {agent.model_name}")

    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        print(f"[+] Đang phân tích ảnh: {args.image}...")
        report = agent.analyze_threat(img, class_name=args.weapon, confidence=0.88)
        print("\n" + "=" * 60)
        print("KẾT QUẢ PHÂN TÍCH TỪ AI AGENT (GEMINI):")
        print("=" * 60)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("[+] Chạy thử nghiệm tự sinh mẫu kiểm tra...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "TEST FRAME (KITCHEN SCENE)", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        report = agent.analyze_threat(dummy_frame, class_name="Knife", confidence=0.92)
        print(json.dumps(report, indent=2, ensure_ascii=False))
