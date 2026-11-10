"""
class_policy.py - Chính sách lớp dùng chung cho TOÀN hệ thống.

VÌ SAO CÓ FILE NÀY
------------------
Trước đây mỗi đường chạy (process_image, process_video, media_scanner,
inference_engine) tự giữ một bảng tên lớp, một bảng màu và một ngưỡng
chung. Hệ quả là khi mô hình đổi từ 3 lớp sang 12 lớp, có file được sửa
có file không, và mọi lớp id >= 3 đều hiện ra là "weapon".

Nay toàn bộ chính sách nằm ở một chỗ duy nhất: khối `class_policy`
trong configs/system_config.yaml. Mọi đường chạy đọc từ đây.

BA Ý TƯỞNG CHÍNH
----------------
1. PHÂN MỨC. Các lớp của dataset không ngang nhau về mức nguy hiểm:
     - "weapon"  : Axe, Sickle, Sword, Knife, Pistol, Long_gun, Grenade
                   -> vũ khí thật, báo động đỏ
     - "suspect" : Crowbar, Stick, Hammer, Spanner, Chain
                   -> dụng cụ, chỉ đánh dấu vàng, không báo động
   Một cái búa trong hộp đồ nghề vẫn là cái búa - mô hình nhận ĐÚNG, nhưng
   người xem gọi đó là báo nhầm. Đây là vấn đề định nghĩa bài toán.

2. NGƯỠNG RIÊNG TỪNG LỚP. Một ngưỡng chung là sai, vì chi phí của hai loại
   lỗi rất khác nhau giữa các lớp:
     - Bỏ sót một khẩu súng là thảm hoạ  -> ngưỡng THẤP (0.30)
     - Báo nhầm một sợi dây xích chỉ gây phiền -> ngưỡng CAO (0.60)

3. LỌC HÌNH HỌC CÓ ĐIỀU KIỆN THEO ĐỘ TIN CẬY.  <-- phần sửa mới
   Bản cũ loại thẳng mọi hộp lớn hơn 12% diện tích khung. Luật đó cắt cả
   vật THẬT ở gần camera, tức là đánh đổi recall để mua precision.
   Bản mới chỉ loại hộp lớn KHI mô hình không chắc. Căn cứ: số liệu chẩn
   đoán trên video thật cho thấy nhiễu bám vào cánh tay / balo nằm ở
   conf 0.10-0.30, còn vật thật ở gần cho conf cao hơn hẳn.

CÁCH DÙNG
---------
    from src.common.class_policy import load_policy

    policy = load_policy()
    conf_floor = policy.predict_conf()                  # truyền vào predict()
    policy.keep("Chain", 0.42)                          # -> False (dưới 0.60)
    policy.keep_geometry(box, W, H, conf=0.81, cls_name="Knife")
    policy.vi_name("Pistol")                            # "SUNG NGAN (PISTOL)"
    policy.color("Pistol")                              # (0, 0, 255) BGR
    policy.tier("Hammer")                               # "suspect"
"""

import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"

# Dùng khi file cấu hình thiếu hoặc hỏng, để hệ thống vẫn chạy được.
_FALLBACK = {
    "detect_conf_floor": 0.15,
    "tiers": {
        "weapon": ["Axe", "Sickle", "Sword", "Knife", "Pistol", "Long_gun", "Grenade"],
        "suspect": ["Crowbar", "Stick", "Hammer", "Spanner", "Chain"],
    },
    "thresholds": {
        "default": 0.40,
        "Pistol": 0.30, "Long_gun": 0.30, "Grenade": 0.30,
        "Axe": 0.35, "Sickle": 0.35, "Sword": 0.40, "Knife": 0.45,
        "Crowbar": 0.50, "Stick": 0.50, "Hammer": 0.55,
        "Spanner": 0.55, "Chain": 0.60,
    },
    "geometry": {
        "tight_area_pct": 12.0,
        "tight_area_pct_suspect": 8.0,
        "max_box_area_pct": 35.0,
        "max_box_area_pct_suspect": 18.0,
        "conf_lo": 0.35,
        "bypass_conf": 0.65,
        "hard_max_box_area_pct": 65.0,
        "min_box_area_pct": 0.02,
        "max_aspect_ratio": 14.0,
    },
    "display": {},
}

_DEFAULT_VI = {
    "Axe": "RIU CHAT (AXE)",
    "Sickle": "LIEM/RUA (SICKLE)",
    "Sword": "KIEM/MA TAU (SWORD)",
    "Crowbar": "XA BENG (CROWBAR)",
    "Knife": "DAO (KNIFE)",
    "Pistol": "SUNG NGAN (PISTOL)",
    "Long_gun": "SUNG TRUONG (LONG GUN)",
    "Grenade": "LUU DAN (GRENADE)",
    "Stick": "GAY (STICK)",
    "Hammer": "BUA (HAMMER)",
    "Spanner": "CO LE (SPANNER)",
    "Chain": "XICH SAT (CHAIN)",
}

_DEFAULT_COLOR = {
    "Axe": (0, 140, 255), "Sickle": (0, 215, 255), "Sword": (255, 0, 128),
    "Crowbar": (205, 90, 106), "Knife": (0, 165, 255), "Pistol": (0, 0, 255),
    "Long_gun": (0, 0, 200), "Grenade": (0, 69, 255), "Stick": (0, 255, 255),
    "Hammer": (30, 105, 210), "Spanner": (180, 105, 255), "Chain": (255, 255, 0),
}

TIER_WEAPON = "weapon"
TIER_SUSPECT = "suspect"
# Màu dự phòng theo mức, dùng khi lớp chưa khai báo màu riêng
TIER_COLOR = {TIER_WEAPON: (0, 0, 255), TIER_SUSPECT: (0, 215, 255)}


class ClassPolicy:
    def __init__(self, cfg: dict):
        self.detect_conf_floor = float(cfg.get("detect_conf_floor", 0.15))

        tiers = cfg.get("tiers") or {}
        self._tier_of = {}
        for tier_name, members in tiers.items():
            for m in (members or []):
                self._tier_of[self._key(m)] = tier_name

        th = dict(cfg.get("thresholds") or {})
        self._default_thresh = float(th.pop("default", 0.40))
        self._thresh = {self._key(k): float(v) for k, v in th.items()}

        geo = cfg.get("geometry") or {}
        # Giới hạn RỘNG: áp dụng khi conf >= bypass_conf.
        self.max_box_area_pct = float(geo.get("max_box_area_pct", 35.0))
        # Nhóm suspect giữ chặt hơn: bỏ sót một cái búa không phải thảm hoạ.
        self.max_box_area_pct_suspect = float(
            geo.get("max_box_area_pct_suspect", min(18.0, self.max_box_area_pct)))
        # Giới hạn CHẶT: áp dụng khi conf <= conf_lo.
        self.tight_area_pct = float(geo.get("tight_area_pct", 12.0))
        self.tight_area_pct_suspect = float(
            geo.get("tight_area_pct_suspect", min(8.0, self.tight_area_pct)))
        # Hai mốc conf dùng để nội suy giữa chặt và rộng.
        self.conf_lo = float(geo.get("conf_lo", 0.35))
        self.bypass_conf = float(geo.get("bypass_conf", 0.65))
        if self.bypass_conf <= self.conf_lo:  # tránh chia cho 0 nếu cấu hình sai
            self.bypass_conf = self.conf_lo + 1e-6
        # Giới hạn CỨNG: trên mức này loại không miễn trừ.
        self.hard_max_box_area_pct = float(geo.get("hard_max_box_area_pct", 65.0))
        self.min_box_area_pct = float(geo.get("min_box_area_pct", 0.02))
        self.max_aspect_ratio = float(geo.get("max_aspect_ratio", 14.0))

        disp = cfg.get("display") or {}
        self._vi = {self._key(k): v for k, v in _DEFAULT_VI.items()}
        self._color = {self._key(k): tuple(v) for k, v in _DEFAULT_COLOR.items()}
        for name, spec in disp.items():
            k = self._key(name)
            if isinstance(spec, dict):
                if spec.get("vi"):
                    self._vi[k] = spec["vi"]
                if spec.get("color"):
                    self._color[k] = tuple(int(c) for c in spec["color"])

    # -- tiện ích nội bộ -------------------------------------------------
    @staticmethod
    def _key(name) -> str:
        """Chuẩn hoá tên lớp: không phân biệt hoa thường và dấu gạch dưới."""
        return str(name).strip().lower().replace("-", "_")

    # -- API ---------------------------------------------------------------
    def tier(self, cls_name) -> str:
        """Mức của lớp. Lớp lạ mặc định coi là vũ khí (an toàn hơn là bỏ qua)."""
        return self._tier_of.get(self._key(cls_name), TIER_WEAPON)

    def is_weapon(self, cls_name) -> bool:
        return self.tier(cls_name) == TIER_WEAPON

    def threshold(self, cls_name) -> float:
        return self._thresh.get(self._key(cls_name), self._default_thresh)

    def keep(self, cls_name, conf) -> bool:
        """Detection có vượt ngưỡng RIÊNG của lớp đó không."""
        return float(conf) >= self.threshold(cls_name)

    def area_limit(self, conf=None, cls_name=None) -> float:
        """
        Giới hạn diện tích CHO PHÉP, nội suy theo độ tin cậy.

        Càng chắc thì càng được phép to. Đây là điểm sửa cốt lõi so với bản
        cũ, nơi giới hạn là một con số cố định 12% áp cho mọi detection.

            conf <= conf_lo      -> tight_area_pct      (chặt)
            conf >= bypass_conf  -> max_box_area_pct    (rộng)
            ở giữa               -> nội suy tuyến tính

        conf = None (lời gọi cũ không truyền conf) -> trả về mức RỘNG, vì
        yêu cầu của hệ thống là không bỏ sót; thiếu thông tin thì nghiêng
        về phía giữ lại.
        """
        suspect = (cls_name is not None and self.tier(cls_name) == TIER_SUSPECT)
        hi = self.max_box_area_pct_suspect if suspect else self.max_box_area_pct
        lo = self.tight_area_pct_suspect if suspect else self.tight_area_pct
        if conf is None:
            return hi
        c = float(conf)
        if c <= self.conf_lo:
            return lo
        if c >= self.bypass_conf:
            return hi
        t = (c - self.conf_lo) / (self.bypass_conf - self.conf_lo)
        return lo + t * (hi - lo)

    def keep_geometry(self, box, frame_w, frame_h, conf=None, cls_name=None):
        """
        Chặn theo HÌNH HỌC của hộp bao. Trả về (giữ?, lý_do_loại).

        `conf` và `cls_name` là tuỳ chọn để không làm vỡ các lời gọi cũ,
        nhưng TRUYỀN VÀO THÌ TỐT HƠN NHIỀU: không có conf thì hàm không
        phân biệt được cánh tay với con dao giơ gần camera.

        Thứ tự xét:
          1. hộp rỗng                      -> loại
          2. nhỏ hơn min_box_area_pct      -> loại (nhiễu điểm, không miễn trừ)
          3. lớn hơn hard_max_box_area_pct -> loại (không miễn trừ)
          4. lớn hơn area_limit(conf, lớp) -> loại
          5. tỉ lệ cạnh bất thường         -> loại, trừ khi conf >= bypass_conf
        """
        if not frame_w or not frame_h:
            return True, None
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            return False, "hộp rỗng"

        area_pct = (w * h) / float(frame_w * frame_h) * 100.0
        c = None if conf is None else float(conf)
        trusted = (c is not None and c >= self.bypass_conf)

        if area_pct < self.min_box_area_pct:
            return False, f"hộp quá nhỏ ({area_pct:.3f}% khung)"

        if area_pct > self.hard_max_box_area_pct:
            return False, (f"hộp vượt giới hạn cứng "
                           f"({area_pct:.1f}% > {self.hard_max_box_area_pct:.0f}%)")

        limit = self.area_limit(c, cls_name)
        if area_pct > limit:
            extra = "" if c is None else f" ở conf {c:.2f}"
            return False, f"hộp quá to ({area_pct:.1f}% > {limit:.1f}% cho phép{extra})"

        ratio = max(w / h, h / w)
        if ratio > self.max_aspect_ratio and not trusted:
            return False, f"tỉ lệ cạnh bất thường ({ratio:.1f} > {self.max_aspect_ratio:.0f})"

        return True, None

    def vi_name(self, cls_name) -> str:
        return self._vi.get(self._key(cls_name), str(cls_name).upper())

    def color(self, cls_name):
        k = self._key(cls_name)
        if k in self._color:
            return self._color[k]
        return TIER_COLOR.get(self.tier(cls_name), (0, 0, 255))

    def min_threshold(self) -> float:
        vals = list(self._thresh.values()) + [self._default_thresh]
        return min(vals) if vals else self._default_thresh

    def predict_conf(self) -> float:
        """
        Ngưỡng truyền vào model.predict().

        Phải THẤP hơn mọi ngưỡng riêng, nếu không lớp có ngưỡng thấp
        (súng, lựu đạn) sẽ bị chặn ngay từ tầng dò và việc đặt ngưỡng
        riêng trở nên vô nghĩa. Đồng thời phải đủ thấp để giai đoạn 2 của
        ByteTrack có detection mờ mà bám vết.
        """
        return min(self.detect_conf_floor, self.min_threshold())

    def filter_detections(self, detections, names, conf_index=4, cls_index=5):
        """
        Lọc danh sách detection dạng [x1, y1, x2, y2, conf, cls_id, ...]
        theo ngưỡng riêng của từng lớp.
        """
        out = []
        for d in detections:
            cls_id = int(d[cls_index])
            cls_name = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) \
                else (names[cls_id] if cls_id < len(names) else f"class_{cls_id}")
            if self.keep(cls_name, d[conf_index]):
                out.append(d)
        return out

    def summary(self) -> str:
        lines = [f"Ngưỡng dò chung: {self.predict_conf():.2f}  |  ngưỡng riêng từng lớp:"]
        by_tier = {}
        for k, t in self._tier_of.items():
            by_tier.setdefault(t, []).append(k)
        for tier_name in (TIER_WEAPON, TIER_SUSPECT):
            members = sorted(by_tier.get(tier_name, []))
            if not members:
                continue
            label = "VŨ KHÍ " if tier_name == TIER_WEAPON else "NGHI VẤN"
            parts = [f"{m}={self._thresh.get(m, self._default_thresh):.2f}" for m in members]
            lines.append(f"  [{label}] " + ", ".join(parts))
        return "\n".join(lines)

    def geometry_summary(self) -> str:
        return (f"Lọc hình học theo conf: ở conf<={self.conf_lo:.2f} cho phép "
                f"{self.tight_area_pct:.0f}% (suspect {self.tight_area_pct_suspect:.0f}%), "
                f"ở conf>={self.bypass_conf:.2f} cho phép {self.max_box_area_pct:.0f}% "
                f"(suspect {self.max_box_area_pct_suspect:.0f}%); "
                f"giới hạn cứng {self.hard_max_box_area_pct:.0f}%; "
                f"nhỏ nhất {self.min_box_area_pct:.2f}%; "
                f"tỉ lệ cạnh <= {self.max_aspect_ratio:.0f}")


def load_full_config(config_path=None) -> dict:
    """Đọc cả file cấu hình. Trả về dict rỗng nếu không đọc được."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if yaml is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # pragma: no cover
        print(f"[CẢNH BÁO] Không đọc được {path}: {e}")
        return {}


def load_inference_cfg(config_path=None) -> dict:
    """
    Tham số suy luận ở mục `inference` của system_config.yaml.

    Có hàm riêng vì bản trước khai iou_threshold và half trong file cấu hình
    nhưng không nơi nào đọc -> thành code chết suốt một thời gian dài.
    """
    cfg = load_full_config(config_path).get("inference") or {}
    return {
        "imgsz": int(cfg.get("imgsz", 960)),
        "conf_threshold": cfg.get("conf_threshold", None),
        "iou": float(cfg.get("iou_threshold", 0.70)),
        "max_det": int(cfg.get("max_det", 100)),
        "agnostic_nms": bool(cfg.get("agnostic_nms", False)),
        "device": cfg.get("device", None),
        "half": bool(cfg.get("half", True)),
        "augment": bool(cfg.get("augment", False)),
        "model_path": cfg.get("model_path", "weights/best.pt"),
    }


def load_tracking_cfg(config_path=None) -> dict:
    """
    Tham số ByteTrack ở mục `tracking` của system_config.yaml.

    Có hàm riêng vì process_video.py từng gán cứng
    WeaponByteTracker(track_high_thresh=0.45, track_low_thresh=0.15)
    và KHÔNG truyền new_track_thresh. Hệ quả: bản sửa new_track_thresh=0.35
    chỉ có hiệu lực ở pipeline camera, còn đường quét video thì không, nên
    vũ khí đang di chuyển ở conf trung bình vẫn không mở được track mới.
    """
    cfg = load_full_config(config_path).get("tracking") or {}
    return {
        "track_high_thresh": float(cfg.get("track_high_thresh", 0.50)),
        "track_low_thresh": float(cfg.get("track_low_thresh", 0.15)),
        "track_buffer": int(cfg.get("track_buffer", 30)),
        "match_thresh": float(cfg.get("match_thresh", 0.80)),
        "new_track_thresh": float(cfg.get("new_track_thresh", 0.35)),
    }


def load_policy(config_path=None) -> ClassPolicy:
    """Nạp chính sách từ configs/system_config.yaml, có bản dự phòng."""
    cfg = load_full_config(config_path).get("class_policy")
    if not cfg:
        print("[CẢNH BÁO] Không tìm thấy khối 'class_policy' -> dùng giá trị dự phòng.")
        cfg = _FALLBACK
    return ClassPolicy(cfg)


if __name__ == "__main__":
    p = load_policy()
    print(p.summary())
    print(p.geometry_summary())
    print()
    for n in ["Pistol", "Knife", "Chain", "Hammer", "Grenade"]:
        print(f"  {n:<10} mức={p.tier(n):<8} ngưỡng={p.threshold(n):.2f}  {p.vi_name(n)}")

    print("\nThử lọc hình học trên khung 1920x1080:")
    W, H = 1920, 1080
    cases = [
        ("dao nhỏ ở xa",          [900, 500, 940, 560],      0.52, "Knife"),
        ("dao giơ gần camera",    [600, 200, 1300, 900],     0.88, "Knife"),
        ("cánh tay bị nhầm",      [600, 200, 1300, 900],     0.22, "Knife"),
        ("balo bị nhầm",          [400, 150, 1500, 1000],    0.19, "Axe"),
        ("búa to conf cao",       [500, 300, 1200, 800],     0.80, "Hammer"),
        ("gần kín khung",         [10, 10, 1900, 1070],      0.95, "Pistol"),
    ]
    for label, box, conf, cls in cases:
        ok, why = p.keep_geometry(box, W, H, conf=conf, cls_name=cls)
        area = (box[2]-box[0])*(box[3]-box[1])/(W*H)*100
        print(f"  {label:<22} {cls:<8} conf={conf:.2f} area={area:5.1f}%  "
              f"-> {'GIỮ' if ok else 'loại: ' + why}")
