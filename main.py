"""
main.py — Streamlit Frontend cho Agricultural AI Chatbot.
Giao diện dark botanical, hiện đại, chuyên nghiệp. Không icon.

=== UPDATES: PESTICIDE INTEGRATION ===
- Thêm "Gợi ý thuốc điều trị" tab trong sidebar
- Sau khi chẩn đoán ảnh, tự động hiển thị card thuốc điều trị
- Cho phép tìm kiếm thuốc theo tên hoạt chất
- PesticideEngine được lazy-loaded như các engine khác

=== FIX: SIDEBAR TOGGLE ===
- Sidebar luôn hiển thị, không cho phép collapse (giống TravelBot)
- Ẩn nút collapsedControl để tránh conflict
- CSS đơn giản, ổn định, không cần JS inject
"""

import os, sys, warnings, time, base64, io
import streamlit as st
from PIL import Image
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GROQ_API_KEY
from translation import detect_language, format_disease_info

# ──────────────────────────────────────────
# QTYPE → NGHIỆP VỤ MAPPING
# ──────────────────────────────────────────
QTYPE_NGHIEP_VU = {
    "Existence & Sanity Check": {
        "vi": "Xác nhận cây trồng trong ảnh",
        "en": "Confirm plant presence in image",
        "instruction_vi": (
            "Xác nhận cây trồng có thực sự hiện diện trong ảnh không. "
            "Kiểm tra ảnh có hợp lệ để phân tích không (ảnh rõ ràng, có lá cây). "
            "Nếu không hợp lệ, hướng dẫn cách chụp ảnh đúng cách."
        ),
        "instruction_en": (
            "Confirm whether the plant is actually present in the image. "
            "Check if the image is valid for analysis (clear, shows leaves). "
            "If not valid, guide how to take a proper photo."
        ),
    },
    "Plant Species Identification": {
        "vi": "Xác định loại cây trồng",
        "en": "Identify plant species",
        "instruction_vi": (
            "Xác định loại cây trồng trong ảnh là gì. "
            "Dựa vào đặc điểm hình thái của lá (hình dạng, màu sắc, gân lá). "
            "Nếu xác định được, cho biết các đặc điểm nhận dạng của loại cây đó."
        ),
        "instruction_en": (
            "Identify what plant species is shown in the image. "
            "Base your answer on leaf morphology (shape, color, veins). "
            "If identified, describe the key characteristics of that plant."
        ),
    },
    "General Health Assessment": {
        "vi": "Đánh giá sức khỏe cây trồng",
        "en": "Assess plant health",
        "instruction_vi": (
            "Đánh giá tổng hợp sức khỏe của cây trồng trong ảnh. "
            "Cây có khỏe mạnh không? Nếu có dấu hiệu bệnh, mức độ nghiêm trọng ra sao? "
            "Đưa ra đánh giá và lời khuyên chăm sóc."
        ),
        "instruction_en": (
            "Give an overall health assessment of the plant in the image. "
            "Is the plant healthy? If there are signs of disease, what is the severity? "
            "Provide assessment and care recommendations."
        ),
    },
    "Visual Attribute Grounding": {
        "vi": "Nhận dạng triệu chứng bệnh",
        "en": "Identify disease symptoms",
        "instruction_vi": (
            "Quan sát và mô tả cụ thể các triệu chứng bệnh nhìn thấy trong ảnh. "
            "Chỉ ra: vị trí triệu chứng (lá, thân, quả), màu sắc thay đổi, "
            "hình dạng tổn thương. Đây là cơ sở để chẩn đoán bệnh chính xác."
        ),
        "instruction_en": (
            "Observe and describe the specific disease symptoms visible in the image. "
            "Point out: symptom location (leaf, stem, fruit), color changes, "
            "lesion patterns. This is the basis for accurate disease diagnosis."
        ),
    },
    "Detailed Verification": {
        "vi": "Xác minh chi tiết bệnh cây",
        "en": "Verify disease details",
        "instruction_vi": (
            "Xác minh chi tiết bệnh cây đã được phân loại. "
            "So sánh các đặc điểm trong ảnh với mô tả bệnh chuẩn. "
            "Đánh giá độ tin cầy của kết quả chẩn đoán và nêu các dấu hiệu đặc trưng."
        ),
        "instruction_en": (
            "Verify the details of the classified disease. "
            "Compare features in the image with standard disease descriptions. "
            "Assess diagnosis reliability and highlight distinguishing signs."
        ),
    },
    "Specific Disease Identification": {
        "vi": "Xác định tên bệnh cụ thể",
        "en": "Identify specific disease",
        "instruction_vi": (
            "Xác định chính xác tên bệnh mà cây trồng đang mắc phải. "
            "Nêu tên bệnh, loại tác nhân gây bệnh (nấm, vi khuẩn, virus). "
            "Cho biết bệnh này phổ biến ở vùng nào và điều kiện thời tiết nào thường xảy ra."
        ),
        "instruction_en": (
            "Precisely identify the disease the plant is suffering from. "
            "State the disease name and type of pathogen (fungal, bacterial, viral). "
            "Indicate which regions and weather conditions this disease commonly occurs in."
        ),
    },
    "Comprehensive Description": {
        "vi": "Mô tả toàn diện về bệnh cây",
        "en": "Comprehensive disease description",
        "instruction_vi": (
            "Mô tả toàn diện về bệnh cây tìm thấy trong ảnh. "
            "Bao gồm: chu trình sinh sản của tác nhân bệnh, giai đoạn lây lẽ, "
            "phạm vi lây nhiễm (chỉ lá hay cả cây), và mức độ thiệt hại kinh tế."
        ),
        "instruction_en": (
            "Provide a comprehensive description of the plant disease found. "
            "Include: pathogen life cycle, infection stages, "
            "spread range (leaf only or whole plant), and economic damage potential."
        ),
    },
    "Causal Reasoning": {
        "vi": "Phân tích nguyên nhân gây bệnh",
        "en": "Analyze disease cause",
        "instruction_vi": (
            "Phân tích nguyên nhân tại sao cây trồng này bị mắc bệnh. "
            "Các yếu tố nào đã tạo điều kiện cho bệnh phát triển? "
            "(thời tiết, độ ẩm, chăm sóc không đúng, lây từ cây khác). "
            "Đưa ra các biện pháp phòng ngừa từ gốc rễ."
        ),
        "instruction_en": (
            "Analyze why this plant got infected with this disease. "
            "What factors created conditions for disease development? "
            "(weather, humidity, improper care, spread from other plants). "
            "Provide root-cause prevention measures."
        ),
    },
    "Counterfactual Reasoning": {
        "vi": "Dự đoán hậu quả nếu không điều trị",
        "en": "Predict consequences without treatment",
        "instruction_vi": (
            "Dự đoán điều gì sẽ xảy ra nếu bệnh này không được điều trị kịp thời. "
            "Bệnh sẽ lây lan như thế nào? Thiệt hại mùa lúa/quả tỉ lệ ra sao? "
            "So sánh: điều trị sớm vs điều trị muộn. "
            "Từ đó đưa ra kế hoạch điều trị khẩn cấp."
        ),
        "instruction_en": (
            "Predict what will happen if this disease is not treated promptly. "
            "How will the disease spread? What percentage of yield/fruit will be lost? "
            "Compare: early treatment vs late treatment outcomes. "
            "Then provide an urgent treatment plan."
        ),
    },
    "Treatment & Pesticide Recommendation": {
        "vi": "Gợi ý thuốc & phác đồ điều trị",
        "en": "Treatment & Pesticide Recommendation",
        "instruction_vi": (
            "Dựa trên kết quả chẩn đoán bệnh, đưa ra phác đồ điều trị đầy đủ. "
            "Bao gồm: tên hoạt chất thuốc, tên sản phẩm thương mại cụ thể, "
            "liều lượng pha chế, cách phun, thời điểm phun tốt nhất, và lưu ý an toàn. "
            "Ưu tiên các sản phẩm từ cơ sở dữ liệu thuốc PPID đã được cung cấp."
        ),
        "instruction_en": (
            "Based on the disease diagnosis, provide a complete treatment protocol. "
            "Include: active ingredient names, specific commercial product names, "
            "mixing rates, application method, optimal timing, and safety precautions. "
            "Prioritize products from the PPID pesticide database provided."
        ),
    },
}


def get_qtype_label(qtype: str, lang: str) -> str:
    info = QTYPE_NGHIEP_VU.get(qtype, {})
    return info.get(lang, qtype)


def get_qtype_instruction(qtype: str, lang: str) -> str:
    info = QTYPE_NGHIEP_VU.get(qtype, {})
    key = f"instruction_{lang}"
    return info.get(
        key,
        f"Trả lời theo loại: {qtype}" if lang == "vi" else f"Answer regarding: {qtype}",
    )


# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="AgriBot — Trợ lý AI Nông Nghiệp",
    page_icon="🌿",
    layout="wide",
    # ✅ FIX: "expanded" để sidebar luôn mở (giống TravelBot)
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA",
        "About": "AgriBot — AI Agricultural Chatbot | Powered by Groq + CLIP + PPID",
    },
)

# ──────────────────────────────────────────
# CSS — Clean Light Theme + Sidebar Always Visible
# ──────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:           #f7f8f5;
    --surface:      #ffffff;
    --surface2:     #f2f4ef;
    --surface3:     #e8ebe3;
    --border:       rgba(0,0,0,0.07);
    --border2:      rgba(0,0,0,0.11);
    --border3:      rgba(0,0,0,0.18);
    --accent:       #3a7d44;
    --accent-dim:   rgba(58,125,68,0.07);
    --accent-mid:   rgba(58,125,68,0.13);
    --accent-str:   rgba(58,125,68,0.25);
    --warn:         #c2540a;
    --warn-dim:     rgba(194,84,10,0.07);
    --info:         #1a6fa3;
    --info-dim:     rgba(26,111,163,0.07);
    --violet:       #6d4fc2;
    --violet-dim:   rgba(109,79,194,0.07);
    --text1:        #1a1e17;
    --text2:        #5c6355;
    --text3:        #9ba594;
    --mono:         'DM Mono', monospace;
    --r:            5px;
    --r2:           9px;
}

* { box-sizing: border-box; }

body, .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text1) !important;
    font-size: 14px;
    line-height: 1.6;
}

/* ════════════════════════════
   FIX SIDEBAR: LUÔN HIỂN THỊ
   Học từ TravelBot (Document 2):
   - Force sidebar display block
   - Ẩn collapsedControl để không thể collapse
════════════════════════════ */

/* ✅ Force sidebar luôn hiển thị, không bị ẩn/collapse */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: none !important;
    min-width: 320px !important;
    width: 320px !important;
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}

/* Đảm bảo nội dung bên trong sidebar cũng đúng width */
[data-testid="stSidebar"] > div:first-child {
    width: 320px !important;
    min-width: 320px !important;
}

/* ✅ Ẩn nút collapse/expand sidebar để tránh người dùng đóng sidebar */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ✅ Ẩn nút collapse bên trong sidebar */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* ════════════════════════════
   HIDE STREAMLIT CHROME
════════════════════════════ */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stAppDeployButton { display: none !important; }

header[data-testid="stHeader"] {
    display: none !important;
}

.stApp, .stApp > div, .main, .main > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"] { background: var(--bg) !important; }

[data-testid="stAppViewContainer"] * { color: inherit; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1120px !important;
    margin: 0 auto;
}

/* ════════════════════════════
   SIDEBAR CONTENT
════════════════════════════ */
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"] { background: var(--surface) !important; }

.sb-label {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text3);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border2);
    margin-bottom: 8px;
    font-family: var(--mono);
}

/* ════════════════════════════
   HEADER
════════════════════════════ */
.agri-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 52px;
    border-bottom: 1px solid var(--border2);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 0 var(--border);
}
.agri-header .wordmark {
    font-size: 14px;
    font-weight: 500;
    color: var(--text1);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.agri-header .wordmark .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
}
.agri-header .tagline {
    font-size: 11px;
    color: var(--text3);
    font-family: var(--mono);
    letter-spacing: 0.04em;
}
.agri-header .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
}
.agri-header .header-badge {
    font-size: 10px;
    padding: 3px 9px;
    border: 1px solid var(--border2);
    border-radius: 3px;
    color: var(--text3);
    font-family: var(--mono);
    letter-spacing: 0.08em;
    background: var(--surface2);
}
.agri-header .online {
    font-size: 10px;
    color: var(--accent);
    font-family: var(--mono);
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 5px;
}
.agri-header .online::before {
    content: '';
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
    display: block;
}

/* ════════════════════════════
   STATS
════════════════════════════ */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px;
    margin-bottom: 2px;
}
.stat-card {
    background: var(--surface2);
    border-radius: var(--r);
    padding: 10px 6px;
    text-align: center;
    border: 1px solid var(--border2);
}
.stat-card .stat-num {
    font-size: 18px;
    font-weight: 400;
    color: var(--accent);
    line-height: 1;
    font-family: var(--mono);
}
.stat-card .stat-label {
    font-size: 8px;
    color: var(--text3);
    margin-top: 4px;
    letter-spacing: 0.08em;
    font-family: var(--mono);
    text-transform: uppercase;
}

/* ════════════════════════════
   CHIPS
════════════════════════════ */
.chip {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.02em;
    border: 1px solid;
    white-space: nowrap;
    font-family: var(--mono);
}
.chip-plant      { background: var(--accent-dim);  color: var(--accent); border-color: var(--accent-str); }
.chip-disease    { background: var(--warn-dim);     color: var(--warn);   border-color: rgba(194,84,10,0.22); }
.chip-healthy    { background: var(--accent-dim);  color: var(--accent); border-color: var(--accent-str); }
.chip-confidence { background: var(--info-dim);    color: var(--info);   border-color: rgba(26,111,163,0.22); }
.chip-qtype      { background: var(--violet-dim);  color: var(--violet); border-color: rgba(109,79,194,0.22); }
.chip-medicine   { background: var(--info-dim);    color: var(--info);   border-color: rgba(26,111,163,0.22); }

/* ════════════════════════════
   DIAGNOSIS / PESTICIDE CARDS
════════════════════════════ */
.diagnosis-card {
    background: var(--accent-dim);
    border: 1px solid var(--accent-str);
    border-left: 2px solid var(--accent);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 10px 14px;
    margin: 0 0 8px;
}
.diagnosis-card.warning {
    background: var(--warn-dim);
    border-color: rgba(194,84,10,0.2);
    border-left-color: var(--warn);
}
.diagnosis-card .card-header {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text2);
    margin-bottom: 7px;
    font-family: var(--mono);
}
.diagnosis-card .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: center;
}

.pesticide-card {
    background: var(--info-dim);
    border: 1px solid rgba(56,189,248,0.15);
    border-left: 2px solid var(--info);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 10px 14px;
    margin: 6px 0 8px;
}
.pesticide-card .card-header {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text2);
    margin-bottom: 7px;
    font-family: var(--mono);
}
.pesticide-card .ingr-chip {
    display: inline-block;
    background: rgba(26,111,163,0.07);
    border: 1px solid rgba(26,111,163,0.2);
    color: var(--info);
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 400;
    margin: 2px;
    font-family: var(--mono);
}
.pesticide-card .prod-row {
    font-size: 12px;
    color: var(--text2);
    line-height: 2;
    margin-top: 6px;
}
.pesticide-card .prod-row strong {
    color: var(--text1);
    font-weight: 500;
}

/* ════════════════════════════
   IMAGE PREVIEW
════════════════════════════ */
.img-preview-wrapper {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--r);
    padding: 10px 12px;
    margin: 0 0 8px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.img-preview-wrapper .img-thumb {
    width: 68px;
    height: 68px;
    border-radius: var(--r);
    object-fit: cover;
    border: 1px solid var(--border3);
    flex-shrink: 0;
}
.img-preview-wrapper .img-info { flex: 1; }
.img-preview-wrapper .img-info .img-label {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 5px;
    font-family: var(--mono);
}
.img-preview-wrapper .img-info .img-tag {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-str);
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 400;
    margin-right: 4px;
    margin-bottom: 3px;
    font-family: var(--mono);
}
.img-preview-wrapper .img-info .img-tag.disease-tag {
    background: var(--warn-dim);
    color: var(--warn);
    border-color: rgba(194,84,10,0.22);
}
.img-preview-wrapper .img-info .img-tag.cached-tag {
    background: var(--info-dim);
    color: var(--info);
    border-color: rgba(26,111,163,0.22);
}

/* ════════════════════════════
   CHAT MESSAGES
════════════════════════════ */
.chat-messages { padding: 28px 32px; }

.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 20px;
    gap: 10px;
    align-items: flex-end;
}
.msg-user .bubble {
    background: var(--accent-dim);
    border: 1px solid var(--accent-str);
    color: var(--text1);
    padding: 12px 16px;
    border-radius: var(--r2) var(--r2) 2px var(--r2);
    max-width: 66%;
    font-size: 13.5px;
    line-height: 1.7;
    font-weight: 300;
}
.msg-user .av {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--text2);
    font-size: 9px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: var(--mono);
    letter-spacing: 0.04em;
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 20px;
    gap: 10px;
    align-items: flex-start;
}
.msg-bot .av {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--accent-mid);
    border: 1px solid var(--accent-str);
    color: var(--accent);
    font-size: 9px;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-family: var(--mono);
    letter-spacing: 0.06em;
}
.msg-bot .bubble {
    background: var(--surface);
    border: 1px solid var(--border2);
    color: var(--text1);
    padding: 14px 18px;
    border-radius: var(--r2) var(--r2) var(--r2) 2px;
    max-width: 80%;
    font-size: 13.5px;
    line-height: 1.8;
    font-weight: 300;
}
.msg-bot .bubble strong { color: var(--accent); font-weight: 500; }

/* ════════════════════════════
   INPUTS
════════════════════════════ */
.stTextInput input {
    border: 1px solid var(--border2) !important;
    border-radius: var(--r) !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
    background: var(--surface) !important;
    color: var(--text1) !important;
    transition: border-color 0.15s, background 0.15s;
}
.stTextInput input:focus {
    border-color: var(--accent-str) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    background: var(--surface) !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: var(--text3) !important; }

/* ════════════════════════════
   BUTTONS
════════════════════════════ */
.stButton button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 12px !important;
    border-radius: var(--r) !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stButton button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}
.stButton button[kind="primary"]:hover {
    background: #2d6636 !important;
    border-color: #2d6636 !important;
    box-shadow: none !important;
}
.stButton button[kind="secondary"] {
    background: var(--surface) !important;
    border-color: var(--border2) !important;
    color: var(--text2) !important;
}
.stButton button[kind="secondary"]:hover {
    background: var(--surface2) !important;
    border-color: var(--border3) !important;
    color: var(--text1) !important;
}

/* ════════════════════════════
   FILE UPLOADER
════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border3) !important;
    border-radius: var(--r) !important;
}
[data-testid="stFileUploader"] label { color: var(--text2) !important; }
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-str) !important;
}

/* ════════════════════════════
   EXPANDER
════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--border2) !important;
    border-radius: var(--r) !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text2) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--border3) !important;
}

/* ════════════════════════════
   ALERTS
════════════════════════════ */
[data-testid="stInfo"] {
    background: var(--info-dim) !important;
    border-color: rgba(26,111,163,0.18) !important;
    color: var(--info) !important;
    font-size: 12px !important;
    border-radius: var(--r) !important;
}
.stAlert {
    border-radius: var(--r) !important;
    font-size: 12px !important;
}

/* ════════════════════════════
   MISC
════════════════════════════ */
hr, .stDivider { border-color: var(--border2) !important; opacity: 1 !important; }
.stCaption { color: var(--text3) !important; font-size: 11px !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface2); }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--border3); }

@media (max-width: 768px) {
    .agri-header { padding: 0 16px; }
    .chat-messages { padding: 16px 18px; }
    .img-preview-wrapper { flex-direction: column; align-items: center; }
    .img-preview-wrapper .img-thumb { width: 100px; height: 100px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
defaults = {
    "messages": [],
    "lang": "vi",
    "groq_client": None,
    "retrieval_engine": None,
    "image_classifier_fixed": None,
    "pesticide_engine": None,
    "df": None,
    "pending_image": None,
    "input_counter": 0,
    "_input_submitted": False,
    "_pending_qtype": None,
    "_cached_classifications": None,
    "_cached_plant": "",
    "_cached_disease": "",
    "_cached_image_b64": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════
# LAZY-LOAD HEAVY OBJECTS
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Đang khởi tạo hệ thống...")
def load_dataset_cached():
    from data_processing import load_dataset

    return load_dataset()


@st.cache_resource(show_spinner="Đang tải mô hình AI...")
def load_retrieval_engine(df):
    from recommendation import RetrievalEngine

    return RetrievalEngine(df)


@st.cache_resource(show_spinner="Đang tải mô hình phân loại ảnh...")
def load_image_classifier():
    from image_classifier_fixed import ImageClassifier

    return ImageClassifier()


@st.cache_resource(show_spinner="Đang kết nối Groq API...")
def load_groq_client():
    from groq_client import GroqClient

    return GroqClient()


@st.cache_resource(show_spinner="Đang tải dữ liệu thuốc PPID...")
def load_pesticide_engine():
    from pesticide_engine import PesticideEngine

    return PesticideEngine()


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def get_lang():
    return st.session_state["lang"]


def _pil_to_base64(img: Image.Image, max_size: int = 150) -> str:
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img_copy.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def add_welcome_message():
    lang = get_lang()
    if lang == "vi":
        msg = (
            "Chào mừng bạn đến với AgriBot.\n\n"
            "Tôi là trợ lý AI chuyên tư vấn nông nghiệp. Tôi có thể giúp bạn:\n\n"
            "Chẩn đoán bệnh cây trồng — Gửi ảnh lá cây hoặc mô tả triệu chứng\n"
            "Gợi ý thuốc điều trị — Từ cơ sở dữ liệu PPID (Canada Health)\n"
            "Trả lời câu hỏi nông nghiệp — Về canh tác, phòng chữa bệnh, phân bón\n"
            "Tra cứu thông tin — Tìm giải pháp từ cơ sở dữ liệu PlantVillage\n\n"
            "Thử gửi ảnh hoặc đặt câu hỏi để bắt đầu."
        )
    else:
        msg = (
            "Welcome to AgriBot.\n\n"
            "I'm your AI agricultural advisor. I can help you with:\n\n"
            "Plant Disease Diagnosis — Upload a leaf image or describe symptoms\n"
            "Pesticide Recommendations — From the PPID database (Canada Health)\n"
            "Agriculture Q&A — About farming, disease prevention, fertilizers\n"
            "Knowledge Search — Find solutions from the PlantVillage database\n\n"
            "Try uploading an image or asking a question to get started."
        )
    st.session_state["messages"].append(
        {
            "role": "bot",
            "content": msg,
            "card_html": "",
            "img_preview_html": "",
            "pesticide_html": "",
        }
    )


# ══════════════════════════════════════════════════════
# BUILD PESTICIDE CARD HTML
# ══════════════════════════════════════════════════════
def _build_pesticide_card_html(disease: str, plant: str, lang: str) -> str:
    engine = st.session_state.get("pesticide_engine")
    if not engine or "healthy" in disease.lower():
        return ""

    try:
        rec = engine.get_treatment_recommendations(disease, plant, lang, top_products=4)
        if rec["is_healthy"] or not rec["active_ingredients"]:
            return ""

        ingr_chips = " ".join(
            f'<span class="ingr-chip">{ing}</span>'
            for ing in rec["active_ingredients"][:5]
        )

        prods_html = ""
        if rec["products"]:
            rows = []
            for p in rec["products"][:4]:
                name = p.get("product_name", "N/A")
                reg = p.get("registration", "N/A")
                ing = p.get("active_ingredient", p.get("ingredient", ""))
                rows.append(f"<strong>{name}</strong> ({ing}) &mdash; Reg: {reg}")
            prods_html = '<div class="prod-row">' + "<br>".join(rows) + "</div>"
        else:
            prods_html = (
                '<div class="prod-row" style="color:var(--text2);">'
                + (
                    "Đang sử dụng tư vấn chuyên gia — xem phản hồi bên dưới."
                    if lang == "vi"
                    else "Using expert advice — see response below."
                )
                + "</div>"
            )

        header = (
            "Gợi ý thuốc · PPID" if lang == "vi" else "Treatment Recommendations · PPID"
        )

        return (
            f'<div class="pesticide-card">'
            f'  <div class="card-header">{header}</div>'
            f'  <div style="margin-bottom:6px;">{ingr_chips}</div>'
            f"  {prods_html}"
            f"</div>"
        )
    except Exception as e:
        print(f"[UI] Pesticide card error: {e}")
        return ""


# ══════════════════════════════════════════════════════
# BUILD IMAGE PREVIEW HTML
# ══════════════════════════════════════════════════════
def _build_image_preview_html(
    img_b64: str,
    plant: str,
    disease: str,
    confidence: float,
    lang: str,
    is_cached: bool = False,
    qtype: str | None = None,
) -> str:
    if not img_b64:
        return ""

    is_healthy = "healthy" in disease.lower()
    label_photo = "Ảnh bạn gửi" if lang == "vi" else "Uploaded image"
    if is_cached:
        label_photo = (
            "Ảnh đã gửi trước đó" if lang == "vi" else "Previously uploaded image"
        )

    plant_tag = f'<span class="img-tag">{plant}</span>'
    if is_healthy:
        status_tag = (
            '<span class="img-tag">Khỏe mạnh</span>'
            if lang == "vi"
            else '<span class="img-tag">Healthy</span>'
        )
    else:
        status_tag = f'<span class="img-tag disease-tag">{disease}</span>'

    conf_label = "chắc chắn" if lang == "vi" else "conf."
    conf_tag = (
        f'<span class="img-tag" style="background:var(--violet-dim);color:var(--violet);'
        f'border-color:rgba(109,79,194,0.22);">{confidence:.1f}% {conf_label}</span>'
    )
    cached_tag = ""
    if is_cached:
        cached_tag = (
            '<span class="img-tag cached-tag">Tiếp tục từ ảnh trước</span>'
            if lang == "vi"
            else '<span class="img-tag cached-tag">From previous image</span>'
        )

    qtype_tag = ""
    if qtype:
        qtype_label = get_qtype_label(qtype, lang)
        qtype_tag = (
            f'<span class="img-tag" style="background:var(--violet-dim);color:var(--violet);'
            f'border-color:rgba(109,79,194,0.22);">{qtype_label}</span>'
        )

    return (
        f'<div class="img-preview-wrapper">'
        f'  <img class="img-thumb" src="data:image/png;base64,{img_b64}" alt="plant leaf" />'
        f'  <div class="img-info">'
        f'    <div class="img-label">{label_photo}</div>'
        f"    <div>{plant_tag}{status_tag}{conf_tag}</div>"
        f'    <div style="margin-top:3px;">{cached_tag}{qtype_tag}</div>'
        f"  </div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════
# BUILD DIAGNOSIS CARD
# ══════════════════════════════════════════════════════
def _build_diagnosis_card(
    classifications: list[dict],
    lang: str,
    qtype: str | None = None,
    is_cached: bool = False,
) -> str:
    if not classifications:
        return ""
    top = classifications[0]
    is_healthy = "healthy" in top["disease"].lower()
    card_cls = "diagnosis-card" if is_healthy else "diagnosis-card warning"
    plant, disease, conf = top["plant"], top["disease"], top["confidence"]

    plant_chip = f'<span class="chip chip-plant">{plant}</span>'
    if is_healthy:
        status_chip = (
            '<span class="chip chip-healthy">Khỏe mạnh</span>'
            if lang == "vi"
            else '<span class="chip chip-healthy">Healthy</span>'
        )
    else:
        status_chip = f'<span class="chip chip-disease">{disease}</span>'

    conf_label = "conf." if lang == "en" else "chắc chắn"
    conf_chip = f'<span class="chip chip-confidence">{conf:.1f}% {conf_label}</span>'

    qtype_chip = ""
    if qtype:
        qtype_label = get_qtype_label(qtype, lang)
        qtype_chip = f'<span class="chip chip-qtype">{qtype_label}</span>'

    med_chip = ""
    if not is_healthy:
        med_chip = (
            '<span class="chip chip-medicine">Xem gợi ý thuốc bên dưới</span>'
            if lang == "vi"
            else '<span class="chip chip-medicine">See drug recommendations below</span>'
        )

    header = (
        ("Tiếp tục phân tích ảnh trước" if is_cached else "Kết quả chẩn đoán")
        if lang == "vi"
        else ("Continuing previous image analysis" if is_cached else "Diagnosis Result")
    )

    return (
        f'<div class="{card_cls}">'
        f'  <div class="card-header">{header}</div>'
        f'  <div class="chip-row">{plant_chip} {status_chip} {conf_chip} {qtype_chip} {med_chip}</div>'
        f"</div>"
    )


# ══════════════════════════════════════════════════════
# MAIN PROCESSING LOGIC
# ══════════════════════════════════════════════════════
def process_query(user_input: str, uploaded_image=None, qtype: str | None = None):
    """Returns (card_html, img_preview_html, pesticide_html, response_text)."""
    lang = get_lang()
    detected = detect_language(user_input)
    if len(user_input.strip()) > 5:
        lang = detected
        st.session_state["lang"] = lang

    engine = st.session_state.get("retrieval_engine")
    classifier = st.session_state.get("image_classifier_fixed")
    groq = st.session_state.get("groq_client")

    # ─── Step 1: Classify image ───────────────────────────────────────────
    image_classifications = []
    detected_plant = ""
    detected_disease = ""
    current_img_b64 = ""

    if uploaded_image is not None and classifier:
        with st.spinner(
            "Đang phân loại ảnh..." if lang == "vi" else "Classifying image..."
        ):
            image_classifications = classifier.classify(uploaded_image, top_k=3)
            # MỚI
            if image_classifications:
                top_result = image_classifications[0]

                # Kiểm tra ảnh không phải thực vật
                if top_result.get("disease") == "NOT_PLANT":
                    error_msg = top_result.get("error", "Ảnh không hợp lệ.")
                    st.warning(f"⚠️ {error_msg}")
                    # Trả về ngay, không classify tiếp
                    err_response = (
                        f"⚠️ **Ảnh không hợp lệ**\n\n{error_msg}\n\n"
                        "**Hướng dẫn chụp ảnh đúng:**\n"
                        "• Chụp gần lá cây, rõ nét\n"
                        "• Đảm bảo lá chiếm > 50% khung hình\n"
                        "• Chụp dưới ánh sáng tự nhiên\n"
                        "• Tránh chụp đất, người, vật dụng khác"
                        if lang == "vi"
                        else f"⚠️ **Invalid image**\n\n{error_msg}\n\n"
                        "**Photo tips:**\n"
                        "• Take a close-up of the leaf, in focus\n"
                        "• Leaf should fill > 50% of frame\n"
                        "• Use natural lighting\n"
                        "• Avoid soil, people, or other objects"
                    )
                    return "", "", "", err_response

                detected_plant = top_result.get("plant", "")
                detected_disease = top_result.get("disease", "")
                st.session_state["_cached_classifications"] = image_classifications
                st.session_state["_cached_plant"] = detected_plant
                st.session_state["_cached_disease"] = detected_disease
                current_img_b64 = _pil_to_base64(uploaded_image)
                st.session_state["_cached_image_b64"] = current_img_b64

    elif uploaded_image is None and qtype is not None:
        cached = st.session_state.get("_cached_classifications")
        if cached:
            image_classifications = cached
            detected_plant = st.session_state.get("_cached_plant", "")
            detected_disease = st.session_state.get("_cached_disease", "")
            current_img_b64 = st.session_state.get("_cached_image_b64", "")

    # ─── Step 2: Retrieval ────────────────────────────────────────────────
    retrieval_results = []
    if engine:
        if image_classifications and qtype:
            with st.spinner("Tìm kiếm..." if lang == "vi" else "Searching..."):
                qtype_results = engine.retrieve_by_question_type(
                    qtype=qtype, plant=detected_plant, top_k=3
                )
                retrieval_results.extend(qtype_results)
                disease_results = engine.retrieve_by_disease(detected_disease, top_k=2)
                seen = {r["Question"] for r in retrieval_results}
                retrieval_results.extend(
                    r for r in disease_results if r["Question"] not in seen
                )
                if not qtype_results:
                    fallback = engine.retrieve_by_question_type(
                        qtype=qtype, plant="", top_k=3
                    )
                    seen = {r["Question"] for r in retrieval_results}
                    retrieval_results.extend(
                        r for r in fallback if r["Question"] not in seen
                    )

        elif image_classifications and not qtype:
            with st.spinner("Tìm kiếm..." if lang == "vi" else "Searching..."):
                enriched_query = f"{detected_disease} {detected_plant} {user_input}"
                retrieval_results = engine.retrieve(enriched_query, top_k=3)
                extra = engine.retrieve_by_disease(detected_disease, top_k=2)
                seen = {r["Question"] for r in retrieval_results}
                retrieval_results.extend(r for r in extra if r["Question"] not in seen)

        elif not image_classifications and qtype:
            with st.spinner("Tìm kiếm..." if lang == "vi" else "Searching..."):
                retrieval_results = engine.retrieve_by_question_type(
                    qtype=qtype, plant="", top_k=5
                )
                if len(retrieval_results) < 3:
                    general = engine.retrieve(user_input, top_k=3)
                    seen = {r["Question"] for r in retrieval_results}
                    retrieval_results.extend(
                        r for r in general if r["Question"] not in seen
                    )
        else:
            with st.spinner("Tìm kiếm..." if lang == "vi" else "Searching..."):
                retrieval_results = engine.retrieve(user_input, top_k=3)

    # ─── Step 3: Build Groq message ───────────────────────────────────────
    is_cached_followup = uploaded_image is None and bool(image_classifications)
    groq_user_message = user_input

    if image_classifications and qtype:
        nghiep_vu_instruction = get_qtype_instruction(qtype, lang)
        nghiep_vu_label = get_qtype_label(qtype, lang)
        context_note = ""
        if is_cached_followup:
            context_note = (
                (
                    f"(Tiếp tục phân tích ảnh đã gửi trước đó.)\n"
                    f"Ảnh đó đã được phân loại: cây {detected_plant}, "
                    f"{'khỏe mạnh' if 'healthy' in detected_disease.lower() else 'bệnh ' + detected_disease}.\n\n"
                )
                if lang == "vi"
                else (
                    f"(Continuing analysis from previously uploaded image.)\n"
                    f"That image was classified as: {detected_plant}, "
                    f"{'healthy' if 'healthy' in detected_disease.lower() else detected_disease}.\n\n"
                )
            )
        else:
            context_note = (
                (
                    f"Ảnh đính kèm đã được phân loại: cây {detected_plant}, "
                    f"{'khỏe mạnh' if 'healthy' in detected_disease.lower() else 'bệnh ' + detected_disease}.\n\n"
                )
                if lang == "vi"
                else (
                    f"The uploaded image is classified as: {detected_plant}, "
                    f"{'healthy' if 'healthy' in detected_disease.lower() else detected_disease}.\n\n"
                )
            )

        if lang == "vi":
            groq_user_message = (
                context_note
                + f"Bạn đang thực hiện nhiệm vụ: **{nghiep_vu_label}**.\n\n"
                f"Yêu cầu cụ thể:\n{nghiep_vu_instruction}\n\n"
                f"Cây trồng: {detected_plant}\n"
                f"Trạng thái: {'Khỏe mạnh' if 'healthy' in detected_disease.lower() else 'Bệnh: ' + detected_disease}\n\n"
                f"Dựa trên thông tin tra cứu và kết quả chẩn đoán ảnh, hãy trả lời theo đúng yêu cầu. "
                f"Đưa ra lời khuyên thực hành cụ thể cho nông dân.\n\nCâu hỏi gốc: {user_input}"
            )
        else:
            groq_user_message = (
                context_note
                + f"You are performing the task: **{nghiep_vu_label}**.\n\n"
                f"Specific requirement:\n{nghiep_vu_instruction}\n\n"
                f"Plant: {detected_plant}\n"
                f"Status: {'Healthy' if 'healthy' in detected_disease.lower() else 'Disease: ' + detected_disease}\n\n"
                f"Based on the reference info and image diagnosis, answer as specified. "
                f"Provide practical advice for farmers.\n\nOriginal question: {user_input}"
            )

    elif not image_classifications and qtype:
        nghiep_vu_instruction = get_qtype_instruction(qtype, lang)
        nghiep_vu_label = get_qtype_label(qtype, lang)
        if lang == "vi":
            groq_user_message = (
                f"Nhiệm vụ: **{nghiep_vu_label}**\n\nYêu cầu:\n{nghiep_vu_instruction}\n\n"
                f"Dựa trên các ví dụ từ cơ sở dữ liệu, hãy trả lời theo yêu cầu. "
                f"Không cần giải thích loại câu hỏi, chỉ cần trả lời theo nghiệp vụ.\n\nYêu cầu gốc: {user_input}"
            )
        else:
            groq_user_message = (
                f"Task: **{nghiep_vu_label}**\n\nRequirement:\n{nghiep_vu_instruction}\n\n"
                f"Based on the knowledge base examples, answer accordingly. "
                f"Do not explain the question type, just answer the task.\n\nOriginal: {user_input}"
            )

    # ─── Step 4: Call Groq ────────────────────────────────────────────────
    response = ""
    if groq:
        history = [
            {
                "role": m["role"] if m["role"] == "user" else "assistant",
                "content": m["content"],
            }
            for m in st.session_state["messages"][-8:]
        ]
        with st.spinner(
            "Đang tạo phản hồi..." if lang == "vi" else "Generating response..."
        ):
            response = groq.chat(
                user_message=groq_user_message,
                lang=lang,
                retrieval_results=retrieval_results,
                image_classifications=image_classifications,
                conversation_history=history,
            )
    else:
        response = "Groq client chưa được khởi tạo. Kiểm tra GROQ_API_KEY."

    # ─── Step 5: Build cards ──────────────────────────────────────────────
    card_html = _build_diagnosis_card(
        image_classifications, lang, qtype=qtype, is_cached=is_cached_followup
    )

    img_preview_html = ""
    if image_classifications and current_img_b64:
        top = image_classifications[0]
        img_preview_html = _build_image_preview_html(
            img_b64=current_img_b64,
            plant=top["plant"],
            disease=top["disease"],
            confidence=top["confidence"],
            lang=lang,
            is_cached=is_cached_followup,
            qtype=qtype,
        )

    pesticide_html = ""
    if (
        image_classifications
        and detected_disease
        and "healthy" not in detected_disease.lower()
    ):
        pesticide_html = _build_pesticide_card_html(
            detected_disease, detected_plant, lang
        )

    return card_html, img_preview_html, pesticide_html, response


# ══════════════════════════════════════════════════════
# RENDER UI
# ══════════════════════════════════════════════════════
def main():
    # ── Init heavy objects ──
    try:
        df = load_dataset_cached()
        st.session_state["df"] = df
    except Exception as e:
        st.sidebar.error(f"Dataset: {e}")
        df = None

    if df is not None and st.session_state["retrieval_engine"] is None:
        try:
            st.session_state["retrieval_engine"] = load_retrieval_engine(df)
        except Exception as e:
            st.sidebar.warning(f"Retrieval: {e}")

    if st.session_state["image_classifier_fixed"] is None:
        try:
            st.session_state["image_classifier_fixed"] = load_image_classifier()
        except Exception as e:
            st.sidebar.warning(f"Image Classifier: {e}")

    if (
        st.session_state["image_classifier_fixed"]
        and df is not None
        and not st.session_state.get("_labels_injected")
    ):
        st.session_state["image_classifier_fixed"].set_labels_from_df(df)
        st.session_state["_labels_injected"] = True

    if st.session_state["groq_client"] is None and GROQ_API_KEY:
        try:
            st.session_state["groq_client"] = load_groq_client()
        except Exception as e:
            st.sidebar.error(f"Groq: {e}")
    elif not GROQ_API_KEY:
        st.sidebar.error("GROQ_API_KEY chưa đặt.")

    if st.session_state["pesticide_engine"] is None:
        try:
            st.session_state["pesticide_engine"] = load_pesticide_engine()
        except Exception as e:
            st.sidebar.warning(f"Pesticide Engine: {e}")

    if not st.session_state["messages"]:
        add_welcome_message()

    # ════════════════════════════════════
    # HEADER
    # ════════════════════════════════════
    st.markdown(
        """
        <div class="agri-header">
            <div class="wordmark">
                <span class="dot"></span>
                AgriBot
                <span class="tagline">Plant Disease Diagnosis &nbsp;&middot;&nbsp; PPID Pesticide Database</span>
            </div>
            <div class="header-right">
                <span class="online">Online</span>
                <span class="header-badge">Groq &middot; CLIP &middot; PPID</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════
    # SIDEBAR
    # ════════════════════════════════════
    with st.sidebar:
        lang = get_lang()

        # Language
        st.markdown('<div class="sb-label">Language</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Tiếng Việt",
                use_container_width=True,
                type="primary" if lang == "vi" else "secondary",
            ):
                st.session_state["lang"] = "vi"
                st.rerun()
        with c2:
            if st.button(
                "English",
                use_container_width=True,
                type="primary" if lang == "en" else "secondary",
            ):
                st.session_state["lang"] = "en"
                st.rerun()
        st.divider()

        # Stats
        ds = len(df) if df is not None else 0
        nd = (
            df["Disease"].nunique()
            if (df is not None and "Disease" in df.columns)
            else 0
        )
        pe = st.session_state.get("pesticide_engine")
        np_prods = pe.get_stats()["n_products"] if pe else 0

        st.markdown(
            f"""
            <div class="stats-row">
                <div class="stat-card">
                    <div class="stat-num">{ds:,}</div>
                    <div class="stat-label">{"Mục QA" if lang == "vi" else "QA"}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{nd}</div>
                    <div class="stat-label">{"Bệnh" if lang == "vi" else "Diseases"}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{np_prods:,}</div>
                    <div class="stat-label">{"Thuốc" if lang == "vi" else "Products"}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Pesticide Search
        st.markdown(
            f'<div class="sb-label">{"Tìm theo hoạt chất" if lang == "vi" else "Search by Ingredient"}</div>',
            unsafe_allow_html=True,
        )
        ingr_search = st.text_input(
            "Ingredient",
            placeholder="azoxystrobin, copper, mancozeb...",
            label_visibility="collapsed",
            key="ingr_search_input",
        )
        if st.button(
            "Tìm kiếm" if lang == "vi" else "Search",
            use_container_width=True,
            type="secondary",
            key="btn_ingr_search",
        ):
            if ingr_search.strip() and pe:
                results = pe.search_by_ingredient(ingr_search.strip(), top_k=8)
                if results:
                    rows_html = ""
                    for r in results:
                        pname = r["product_name"]
                        ingr = r["active_ingredient"]
                        reg = r["registration"]
                        rows_html += f"""
                        <div style="border-bottom:1px solid var(--border);padding:8px 0;line-height:1.6;">
                            <div style="font-size:12px;font-weight:500;color:var(--text1);">{pname}</div>
                            <div style="font-size:11px;color:var(--text2);margin-top:1px;font-family:var(--mono);">
                                {ingr} &nbsp;&middot;&nbsp; {reg}
                            </div>
                        </div>"""
                    st.markdown(
                        f'<div style="background:var(--surface2);border:1px solid var(--border2);'
                        f'border-radius:var(--r);padding:10px 12px;max-height:280px;overflow-y:auto;">'
                        f'<div style="font-size:9px;font-weight:500;letter-spacing:.14em;text-transform:uppercase;'
                        f'font-family:var(--mono);color:var(--accent);margin-bottom:8px;">'
                        f"{'Tìm thấy' if lang == 'vi' else 'Found'} {len(results)} {'sản phẩm' if lang == 'vi' else 'products'}</div>"
                        f"{rows_html}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        f"Không tìm thấy sản phẩm cho '{ingr_search}'."
                        if lang == "vi"
                        else f"No products found for '{ingr_search}'."
                    )
            elif not pe:
                st.warning(
                    "Pesticide engine chưa sẵn sàng."
                    if lang == "vi"
                    else "Pesticide engine not ready."
                )
        st.divider()

        # Quick Questions
        st.markdown(
            f'<div class="sb-label">{"Câu hỏi nhanh" if lang == "vi" else "Quick Questions"}</div>',
            unsafe_allow_html=True,
        )
        qs_vi = [
            "Bệnh cà chua thường gặp là gì?",
            "Cách chữa bệnh ghẻ táo?",
            "Bệnh héo lá ngô là do gì?",
            "Phòng bệnh khoai tây như thế nào?",
            "Các loại bệnh nho phổ biến?",
            "Thuốc nào chữa bệnh phấn trắng?",
        ]
        qs_en = [
            "What are common tomato diseases?",
            "How to treat apple scab?",
            "What causes corn leaf blight?",
            "How to prevent potato diseases?",
            "Common grape diseases?",
            "What pesticide treats powdery mildew?",
        ]
        for i, q in enumerate(qs_vi if lang == "vi" else qs_en):
            if st.button(
                q, use_container_width=True, type="secondary", key=f"quick_{i}"
            ):
                st.session_state["_quick_q"] = q
                st.session_state["_pending_qtype"] = None
                st.rerun()
        st.divider()

        # Question Type / Analysis Mode
        qt_title = "Chế độ phân tích" if lang == "vi" else "Analysis Mode"
        with st.expander(qt_title, expanded=False):
            qt_map = {
                "Existence & Sanity Check": ("Xác nhận cây trong ảnh", "Confirm Plant"),
                "Plant Species Identification": (
                    "Xác định loại cây",
                    "Identify Species",
                ),
                "General Health Assessment": (
                    "Đánh giá sức khỏe cây",
                    "Health Assessment",
                ),
                "Visual Attribute Grounding": (
                    "Nhận dạng triệu chứng",
                    "Identify Symptoms",
                ),
                "Detailed Verification": ("Xác minh chi tiết bệnh", "Verify Details"),
                "Specific Disease Identification": (
                    "Xác định tên bệnh",
                    "Identify Disease",
                ),
                "Comprehensive Description": (
                    "Mô tả toàn diện bệnh",
                    "Full Description",
                ),
                "Causal Reasoning": ("Phân tích nguyên nhân bệnh", "Analyze Cause"),
                "Counterfactual Reasoning": (
                    "Dự đoán nếu không điều trị",
                    "Predict Without Treatment",
                ),
                "Treatment & Pesticide Recommendation": (
                    "Gợi ý thuốc điều trị",
                    "Recommend Treatment",
                ),
            }

            has_pending = st.session_state.get("pending_image") is not None
            has_cached = st.session_state.get("_cached_classifications") is not None
            has_any_image_context = has_pending or has_cached

            if has_pending:
                st.info(
                    "Bạn đang có ảnh chờ phân tích."
                    if lang == "vi"
                    else "You have a pending image."
                )
            elif has_cached:
                cached_plant = st.session_state.get("_cached_plant", "")
                cached_disease = st.session_state.get("_cached_disease", "")
                st.info(
                    f"Ảnh: {cached_plant} — "
                    f"{'Khỏe mạnh' if 'healthy' in cached_disease.lower() else cached_disease}."
                    if lang == "vi"
                    else f"Image: {cached_plant} — "
                    f"{'Healthy' if 'healthy' in cached_disease.lower() else cached_disease}."
                )
            else:
                st.caption(
                    "Gửi ảnh lá cây trước để phân tích chuyên sâu."
                    if lang == "vi"
                    else "Upload a leaf image first for in-depth analysis."
                )

            for raw, (vi_l, en_l) in qt_map.items():
                if st.button(
                    vi_l if lang == "vi" else en_l,
                    use_container_width=True,
                    type="secondary",
                    key=f"qt_{raw}",
                ):
                    st.session_state["_pending_qtype"] = raw
                    nghiep_vu = get_qtype_label(raw, lang)
                    q = (
                        (
                            f"Phân tích ảnh theo: {nghiep_vu}"
                            if lang == "vi"
                            else f"Analyze image for: {nghiep_vu}"
                        )
                        if has_any_image_context
                        else (
                            f"Cho tôi thông tin về: {nghiep_vu}"
                            if lang == "vi"
                            else f"Tell me about: {nghiep_vu}"
                        )
                    )
                    st.session_state["_quick_q"] = q
                    st.rerun()
        st.divider()

        # Image Upload
        st.markdown(
            f'<div class="sb-label">{"Ảnh bệnh lá" if lang == "vi" else "Leaf Image"}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.get("_cached_classifications") and not st.session_state.get(
            "pending_image"
        ):
            cached_plant = st.session_state.get("_cached_plant", "")
            cached_disease = st.session_state.get("_cached_disease", "")
            st.markdown(
                f'<p style="font-size:11px;color:var(--text2);margin:0 0 6px;font-family:var(--mono);">'
                f"{'Ảnh hiện tại' if lang == 'vi' else 'Current'}: "
                f"<span style='color:var(--text1)'>{cached_plant}</span> — "
                f"{'Khỏe mạnh' if 'healthy' in cached_disease.lower() else cached_disease}</p>",
                unsafe_allow_html=True,
            )
            reset_lbl = "Xóa ảnh" if lang == "vi" else "Reset image"
            if st.button(
                reset_lbl,
                type="secondary",
                use_container_width=True,
                key="btn_reset_img",
            ):
                st.session_state["_cached_classifications"] = None
                st.session_state["_cached_plant"] = ""
                st.session_state["_cached_disease"] = ""
                st.session_state["_cached_image_b64"] = ""
                st.rerun()

        uploaded_file = st.file_uploader(
            "Chọn ảnh lá cây..." if lang == "vi" else "Choose a leaf image...",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(
                img,
                caption="Ảnh đã chọn" if lang == "vi" else "Selected image",
                use_container_width=True,
            )
            st.session_state["pending_image"] = img
            btn_lbl = "Chẩn đoán ảnh này" if lang == "vi" else "Diagnose this image"
            if st.button(
                btn_lbl, type="primary", use_container_width=True, key="btn_diagnose"
            ):
                q = (
                    "Chẩn đoán bệnh cây trong ảnh này"
                    if lang == "vi"
                    else "Diagnose the plant disease in this image"
                )
                st.session_state["_quick_q"] = q
                st.session_state["_pending_qtype"] = None
                st.rerun()
        else:
            if "_quick_q" not in st.session_state:
                st.session_state["pending_image"] = None
        st.divider()

        # Clear chat
        st.markdown(
            f'<div class="sb-label">{"Điều hướng" if lang == "vi" else "Navigation"}</div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Xóa lịch sử chat" if lang == "vi" else "Clear chat history",
            use_container_width=True,
            type="secondary",
            key="btn_clear",
        ):
            st.session_state["messages"] = []
            st.session_state["pending_image"] = None
            st.session_state["_pending_qtype"] = None
            st.session_state["_cached_classifications"] = None
            st.session_state["_cached_plant"] = ""
            st.session_state["_cached_disease"] = ""
            st.session_state["_cached_image_b64"] = ""
            add_welcome_message()
            st.rerun()

    # ════════════════════════════════════
    # CHAT AREA
    # ════════════════════════════════════
    with st.columns([1])[0]:
        msg_ph = st.empty()

        def render_all():
            with msg_ph.container():
                st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
                for m in st.session_state["messages"]:
                    if m["role"] == "user":
                        img_note = (
                            " &nbsp;<em style='font-size:11px;color:var(--text3);font-style:normal;'>(+ ảnh)</em>"
                            if m.get("has_image")
                            else ""
                        )
                        st.markdown(
                            f"""
                            <div class="msg-user">
                                <div class="bubble" style="white-space:pre-wrap;">{m["content"]}{img_note}</div>
                                <div class="av">YOU</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        card = m.get("card_html", "")
                        img_preview = m.get("img_preview_html", "")
                        pesticide_html = m.get("pesticide_html", "")
                        st.markdown(
                            f"""
                            <div class="msg-bot">
                                <div class="av">AI</div>
                                <div class="bubble" style="white-space:pre-wrap;">{img_preview}{card}{pesticide_html}{m["content"]}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

        render_all()

        # Input row
        lang = get_lang()
        quick_q = st.session_state.pop("_quick_q", None)
        qtype = st.session_state.pop("_pending_qtype", None)
        inp_key = f"user_input_{st.session_state['input_counter']}"

        def _on_change():
            st.session_state["_input_submitted"] = True

        ic = st.columns([5, 1])
        ph = (
            "Hỏi về bệnh cây, thuốc điều trị, canh tác..."
            if lang == "vi"
            else "Ask about diseases, pesticides, farming..."
        )
        with ic[0]:
            user_input = st.text_input(
                "Input",
                placeholder=ph,
                label_visibility="collapsed",
                key=inp_key,
                on_change=_on_change,
            )
        with ic[1]:
            send_clicked = st.button(
                "Gửi" if lang == "vi" else "Send",
                type="primary",
                use_container_width=True,
                key="btn_send",
            )

        final = None
        if quick_q:
            final = quick_q
        elif send_clicked and user_input.strip():
            final = user_input.strip()
        elif st.session_state.get("_input_submitted") and user_input.strip():
            final = user_input.strip()
            st.session_state["_input_submitted"] = False

        if final:
            has_fresh_img = st.session_state.get("pending_image") is not None
            has_cached_img = (
                st.session_state.get("_cached_classifications") is not None
                and qtype is not None
            )
            has_img = has_fresh_img or has_cached_img

            st.session_state["messages"].append(
                {"role": "user", "content": final, "has_image": has_img}
            )

            pending = st.session_state.get("pending_image")
            card_html, img_preview_html, pesticide_html, resp_text = process_query(
                final, uploaded_image=pending, qtype=qtype
            )

            st.session_state["messages"].append(
                {
                    "role": "bot",
                    "content": resp_text,
                    "card_html": card_html,
                    "img_preview_html": img_preview_html,
                    "pesticide_html": pesticide_html,
                }
            )

            st.session_state["pending_image"] = None
            st.session_state["_pending_qtype"] = None
            st.session_state["_input_submitted"] = False
            st.session_state["input_counter"] += 1
            st.rerun()

        hint = (
            "Gửi ảnh lá cây để chẩn đoán tự động và nhận gợi ý thuốc từ PPID."
            if lang == "vi"
            else "Upload a leaf image for automatic diagnosis and pesticide recommendations from PPID."
        )
        st.markdown(
            f'<p style="text-align:center;color:var(--text3);font-size:11px;'
            f'margin-top:10px;letter-spacing:0.04em;font-family:var(--mono);">{hint}</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
