"""
image_classifier_fixed.py — Phân loại bệnh cây trồng từ ảnh.

=== STREAMLIT CLOUD COMPATIBLE VERSION ===

Thay đổi kiến trúc:
  TRƯỚC: CLIP local (~600MB model, cần torch nặng, hay lỗi khi deploy)
  SAU:   Groq Vision API (meta-llama/llama-4-scout-17b-16e-instruct) — gọi API, không cần download model

Ưu điểm:
  ✅ Không cần CLIP, open-clip-torch, git install
  ✅ Deploy Streamlit Cloud không lỗi
  ✅ Tính năng kéo ảnh chẩn đoán vẫn hoạt động bình thường
  ✅ main.py không cần sửa — interface classify() giữ nguyên
"""

import os
import sys
import base64
import warnings
import io
import colorsys
import json
import re

warnings.filterwarnings("ignore")

from PIL import Image

# ── Torch là optional ────────────────────────────────────────────────────────
try:
    import torch

    TORCH_AVAILABLE = True
    print(f"[IMG] ✅ Torch available (dùng cho EfficientNet nếu có model .pth)")
except Exception:
    TORCH_AVAILABLE = False
    print(f"[IMG] ℹ️ Torch không có — dùng Groq Vision thay thế (bình thường)")

from config import LABEL_ENCODER_PATH, IMAGES_DIR, GROQ_API_KEY


# ══════════════════════════════════════════════════════════════════
# PLANT IMAGE CHECK — color heuristic, không cần model
# ══════════════════════════════════════════════════════════════════


def _is_plant_image(img: Image.Image) -> tuple[bool, str]:
    """Kiểm tra ảnh có màu thực vật không (xanh lá hoặc nâu vàng)."""
    img_small = img.convert("RGB").resize((64, 64))
    pixels = list(img_small.getdata())
    total = len(pixels)

    green_count = brown_count = 0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h_deg = h * 360
        if 60 <= h_deg <= 160 and s > 0.15 and v > 0.1:
            green_count += 1
        elif 15 <= h_deg < 60 and s > 0.2 and v > 0.15:
            brown_count += 1

    plant_ratio = (green_count + brown_count) / total

    if plant_ratio >= 0.10:
        return True, ""
    return False, (
        f"Ảnh này không có dấu hiệu của lá cây hoặc cây trồng "
        f"(chỉ {plant_ratio * 100:.1f}% màu thực vật). "
        f"Vui lòng gửi ảnh chụp rõ lá cây hoặc bộ phận cây trồng."
    )


# ══════════════════════════════════════════════════════════════════
# ẢNH → BASE64
# ══════════════════════════════════════════════════════════════════


def _pil_to_base64(img: Image.Image, max_size: int = 448) -> str:
    """Resize và encode ảnh thành base64 JPEG để gửi API."""
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ══════════════════════════════════════════════════════════════════
# FALLBACK LABELS
# ══════════════════════════════════════════════════════════════════

FALLBACK_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour_cherry)___Powdery_mildew",
    "Cherry_(including_sour_cherry)___healthy",
    "Corn_(Maize)___Cercospora_leaf_blight_Gray_leaf_spot",
    "Corn_(Maize)___Common_rust_",
    "Corn_(Maize)___Northern_Leaf_Blight",
    "Corn_(Maize)___healthy",
    "Grape___Black_rot",
    "Grape___Downy_mildew",
    "Grape___Leaf_scorch",
    "Grape___healthy",
    "Orange___Huanglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_Bell___Bacterial_spot",
    "Pepper,_Bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_(Two-spotted_spider_mite)",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ══════════════════════════════════════════════════════════════════
# GROQ VISION PROMPT
# ══════════════════════════════════════════════════════════════════

VISION_PROMPT = """You are an expert plant pathologist. Analyze this plant leaf image.

Identify:
1. Plant species (Tomato, Potato, Apple, Grape, Corn, Pepper, Strawberry, Cherry, Peach, Orange, Blueberry, Raspberry)
2. Health status: healthy or diseased
3. If diseased, the specific disease

Known diseases:
Fungal: Late blight, Early blight, Powdery mildew, Downy mildew, Apple scab, Black rot,
Cedar apple rust, Leaf Mold, Septoria leaf spot, Target Spot, Gray leaf spot,
Northern Leaf Blight, Cercospora leaf blight, Leaf scorch, Common rust
Bacterial: Bacterial spot
Viral: Tomato mosaic virus, Tomato Yellow Leaf Curl Virus
Pest: Spider mites
Other: Huanglongbing (Citrus greening)

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "plant": "plant name",
  "disease": "disease name or healthy",
  "is_healthy": true or false,
  "confidence": number 0-100,
  "symptoms": "brief visible symptoms description"
}

If NOT a plant image:
{"plant": "Unknown", "disease": "NOT_PLANT", "is_healthy": false, "confidence": 0, "symptoms": "Not a plant"}
"""


# ══════════════════════════════════════════════════════════════════
# IMAGE CLASSIFIER — dùng Groq Vision API
# ══════════════════════════════════════════════════════════════════


class ImageClassifier:
    """
    Plant disease classifier dùng Groq Vision API.
    Interface giống hệt CLIP classifier cũ → main.py không cần sửa.
    """

    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self):
        self.labels = self._load_labels()
        self._client = None
        self._init_groq()
        print("[IMG] ✅ ImageClassifier (Groq Vision) ready.")

    # ──────────────────────────────────────
    def _init_groq(self):
        if not GROQ_API_KEY:
            print("[IMG] ⚠️ GROQ_API_KEY chưa đặt — image classification disabled.")
            return
        try:
            from groq import Groq

            self._client = Groq(api_key=GROQ_API_KEY)
            print(f"[IMG] ✅ Groq Vision client ready ({self.VISION_MODEL})")
        except Exception as e:
            print(f"[IMG] ❌ Groq init failed: {e}")
            self._client = None

    # ──────────────────────────────────────
    def _load_labels(self) -> list[str]:
        try:
            import joblib

            if os.path.exists(LABEL_ENCODER_PATH):
                le = joblib.load(LABEL_ENCODER_PATH)
                labels = list(le.classes_)
                print(f"[IMG] Loaded {len(labels)} labels.")
                return labels
        except Exception as e:
            print(f"[IMG] ⚠️ LabelEncoder error: {e}")
        return FALLBACK_LABELS

    # ──────────────────────────────────────
    def set_labels_from_df(self, df):
        """Tương thích interface cũ."""
        if "Plant" not in df.columns or "Disease" not in df.columns:
            return
        pairs = (df["Plant"].astype(str) + "___" + df["Disease"].astype(str)).unique()
        labels = [p for p in pairs if "Unknown" not in p]
        if labels:
            self.labels = sorted(set(labels))
            print(f"[IMG] Labels updated: {len(self.labels)} classes.")

    # ──────────────────────────────────────
    @staticmethod
    def _parse_label(label: str):
        """'Tomato___Late_blight' → ('Tomato', 'Late blight', False)"""
        if "___" in label:
            plant, disease = label.split("___", 1)
        else:
            plant, disease = label, label
        plant = plant.replace("_", " ").replace("(", "").replace(")", "").strip()
        disease = disease.replace("_", " ").replace("(", "").replace(")", "").strip()
        return plant, disease, "healthy" in disease.lower()

    # ──────────────────────────────────────
    def _call_groq_vision(self, img_b64: str) -> dict | None:
        """Gọi Groq Vision API, parse JSON response."""
        try:
            response = self._client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {"type": "text", "text": VISION_PROMPT},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            # Bỏ markdown fences nếu có
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            # Tìm JSON object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            print(f"[IMG] ⚠️ Không tìm được JSON trong response: {raw[:150]}")
            return None

        except Exception as e:
            print(f"[IMG] ⚠️ Groq Vision error: {e}")
            return None

    # ──────────────────────────────────────
    def _map_to_known_label(self, plant: str, disease: str) -> tuple[str, str]:
        """
        Fuzzy-map kết quả Groq về label chuẩn trong dataset.
        Ví dụ: "tomato" + "late blight" → "Tomato" + "Late blight"
        """
        p_low = plant.lower().strip()
        d_low = disease.lower().strip()

        best_label = None
        best_score = 0

        for label in self.labels:
            lp, ld, _ = self._parse_label(label)
            lp_l = lp.lower()
            ld_l = ld.lower()

            score = 0
            # Plant match
            if p_low in lp_l or lp_l in p_low:
                score += 2
            elif any(w in lp_l for w in p_low.split() if len(w) > 3):
                score += 1
            # Disease match
            if d_low in ld_l or ld_l in d_low:
                score += 3
            elif any(w in ld_l for w in d_low.split() if len(w) > 3):
                score += 1

            if score > best_score:
                best_score = score
                best_label = (lp, ld)

        if best_label and best_score >= 2:
            return best_label
        return plant.title(), disease  # capitalize nếu không match

    # ──────────────────────────────────────
    def classify(self, image_path_or_pil, top_k: int = 3) -> list[dict]:
        """
        Phân loại ảnh bệnh cây.
        Returns list[dict]: label, plant, disease, confidence, raw_score
        """
        # Load ảnh
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert("RGB")
        else:
            img = image_path_or_pil.convert("RGB")

        # Bước 1: Kiểm tra màu sắc nhanh
        is_plant, reason = _is_plant_image(img)
        if not is_plant:
            return [
                {
                    "label": "NOT_PLANT",
                    "plant": "Không xác định",
                    "disease": "NOT_PLANT",
                    "confidence": 0.0,
                    "raw_score": 0.0,
                    "error": reason,
                }
            ]

        # Bước 2: Groq Vision không available
        if self._client is None:
            return [
                {
                    "label": "Unknown___Unknown",
                    "plant": "Unknown",
                    "disease": "Chưa thể phân loại — kiểm tra GROQ_API_KEY",
                    "confidence": 0.0,
                    "raw_score": 0.0,
                }
            ]

        # Bước 3: Encode ảnh → base64
        img_b64 = _pil_to_base64(img, max_size=448)

        # Bước 4: Gọi Groq Vision
        result = self._call_groq_vision(img_b64)

        # Bước 5: Xử lý kết quả
        if result is None:
            return [
                {
                    "label": "Unknown___Unknown",
                    "plant": "Unknown",
                    "disease": "Lỗi phân tích ảnh, vui lòng thử lại",
                    "confidence": 0.0,
                    "raw_score": 0.0,
                }
            ]

        # NOT_PLANT từ Groq
        if result.get("disease") == "NOT_PLANT":
            return [
                {
                    "label": "NOT_PLANT",
                    "plant": "Không xác định",
                    "disease": "NOT_PLANT",
                    "confidence": 0.0,
                    "raw_score": 0.0,
                    "error": "Ảnh không phải lá cây, vui lòng chụp lại.",
                }
            ]

        # Parse
        plant = str(result.get("plant", "Unknown")).strip()
        disease = str(result.get("disease", "Unknown")).strip()
        confidence = float(result.get("confidence", 70))
        is_healthy = bool(result.get("is_healthy", False))
        symptoms = str(result.get("symptoms", ""))

        if is_healthy:
            disease = "healthy"

        # Map về label chuẩn
        plant_m, disease_m = self._map_to_known_label(plant, disease)
        label_str = f"{plant_m}___{disease_m}".replace(" ", "_")

        results = [
            {
                "label": label_str,
                "plant": plant_m,
                "disease": disease_m,
                "confidence": confidence,
                "raw_score": confidence,
                "symptoms": symptoms,
            }
        ]

        # Thêm healthy alternative nếu confidence thấp
        if not is_healthy and confidence < 80 and top_k > 1:
            results.append(
                {
                    "label": f"{plant_m}___healthy".replace(" ", "_"),
                    "plant": plant_m,
                    "disease": "healthy",
                    "confidence": max(5.0, 100 - confidence - 15),
                    "raw_score": max(5.0, 100 - confidence - 15),
                    "symptoms": "",
                }
            )

        return results[:top_k]

    # ──────────────────────────────────────
    def get_plants(self) -> list[str]:
        plants = set()
        for label in self.labels:
            plant, _, _ = self._parse_label(label)
            plants.add(plant)
        return sorted(plants)


# ══════════════════════════════════════════════════════════════════
# EFFICIENTNET CLASSIFIER (dùng khi có model .pth từ train local)
# ══════════════════════════════════════════════════════════════════


class EfficientNetClassifier:
    """
    EfficientNet-B0 fine-tuned — chỉ hoạt động khi có file model .pth.
    Trên Streamlit Cloud thường không có → fallback về ImageClassifier.
    """

    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "artifacts",
        "efficientnet_plantvillage.pth",
    )
    CLASS_MAP_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "artifacts", "class_to_idx.pkl"
    )

    def __init__(self):
        self.model = None
        self.class_names = []
        self.device = None
        self.transform = None
        self._load()

    def is_available(self) -> bool:
        return self.model is not None and len(self.class_names) > 0 and TORCH_AVAILABLE

    def _load(self):
        if not TORCH_AVAILABLE or not os.path.exists(self.MODEL_PATH):
            return
        try:
            import joblib, torch.nn as nn
            from torchvision import models, transforms

            class_to_idx = joblib.load(self.CLASS_MAP_PATH)
            self.class_names = [
                k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])
            ]
            num_classes = len(self.class_names)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            model = models.efficientnet_b0(weights=None)
            in_f = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(in_f, num_classes)
            )
            model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            model.eval()
            self.model = model.to(self.device)

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            print(
                f"[EFFNET] ✅ EfficientNet loaded | {num_classes} classes | {self.device}"
            )
        except Exception as e:
            print(f"[EFFNET] ❌ Load thất bại: {e}")
            self.model = None

    @staticmethod
    def _parse_class_name(cn: str):
        if "___" in cn:
            p, d = cn.split("___", 1)
        else:
            p, d = cn, cn
        return p.replace("_", " ").strip(), d.replace("_", " ").strip()

    def classify(self, image_path_or_pil, top_k: int = 3) -> list[dict]:
        if not self.is_available():
            return []
        img = (
            Image.open(image_path_or_pil).convert("RGB")
            if isinstance(image_path_or_pil, str)
            else image_path_or_pil.convert("RGB")
        )
        is_plant, reason = _is_plant_image(img)
        if not is_plant:
            return [
                {
                    "label": "NOT_PLANT",
                    "plant": "Không xác định",
                    "disease": "NOT_PLANT",
                    "confidence": 0.0,
                    "raw_score": 0.0,
                    "error": reason,
                }
            ]
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).squeeze(0).cpu().numpy()
        results = []
        for idx in probs.argsort()[-top_k:][::-1]:
            plant, disease = self._parse_class_name(self.class_names[idx])
            results.append(
                {
                    "label": self.class_names[idx],
                    "plant": plant,
                    "disease": disease,
                    "confidence": float(probs[idx]) * 100,
                    "raw_score": float(probs[idx]) * 100,
                }
            )
        return results
