import os
import json
import re
import unicodedata

import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Cấu hình Streamlit
# =========================
st.set_page_config(
    page_title="VSMEC Demo - PhoBERT / XLM-R / mBERT",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Demo phân loại cảm xúc VSMEC")
st.write(
    "Nhập câu tiếng Việt, 3 mô hình PhoBERT, XLM-R, mBERT (đã fine-tune) "
    "sẽ cùng dự đoán sắc thái cảm xúc."
)

# =========================
# Tiền xử lý + tách từ
# =========================

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MULTISPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = URL_RE.sub(" <url> ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

# underthesea cho PhoBERT
try:
    from underthesea import word_tokenize as uts_word_tokenize
    USE_UTS = True
    uts_error = None
except Exception as e:
    USE_UTS = False
    uts_error = str(e)

def underthesea_segment(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text
    if USE_UTS:
        try:
            seg = uts_word_tokenize(text, format="text")
            if isinstance(seg, list):
                seg = " ".join(seg)
            return MULTISPACE_RE.sub(" ", seg).strip()
        except Exception:
            pass
    # fallback: tách theo khoảng trắng
    return " ".join(text.split())


# =========================
# Load model từ thư mục local
# =========================

@st.cache_resource
def load_model_bundle(model_dir: str, device: str, use_fast: bool = True):
    """
    Load model + tokenizer + labels từ thư mục model_dir.
    Trả về dict: {model, tokenizer, id2label, label_list, device}
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    labels_path = os.path.join(model_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            label_info = json.load(f)
        label_list = label_info.get("label_list")
        id2label = label_info.get("id2label")
        if id2label is not None:
            id2label = {int(k): v for k, v in id2label.items()}
        else:
            id2label = model.config.id2label
            label_list = [id2label[i] for i in range(model.config.num_labels)]
    else:
        id2label = model.config.id2label
        label_list = [id2label[i] for i in range(model.config.num_labels)]

    return {
        "model": model,
        "tokenizer": tokenizer,
        "id2label": id2label,
        "label_list": label_list,
        "device": device,
        "dir": model_dir,
    }


# =========================
# Hàm dự đoán 1 câu
# =========================

@torch.inference_mode()
def predict_one(text: str, bundle, max_length: int = 128, use_seg: bool = False):
    """
    Dự đoán 1 câu với 1 model.
    use_seg=True -> dùng underthesea_segment, ngược lại dùng normalize_text.
    """
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    id2label = bundle["id2label"]
    device = bundle["device"]
    label_list = bundle["label_list"]

    if use_seg:
        processed = underthesea_segment(text)
    else:
        processed = normalize_text(text)

    if not processed:
        return {
            "input": text,
            "processed": processed,
            "pred_label": None,
            "pred_conf": 0.0,
            "probs": {lab: 0.0 for lab in label_list},
        }

    enc = tokenizer(
        processed,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = id2label[pred_id]
    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "input": text,
        "processed": processed,
        "pred_label": pred_label,
        "pred_conf": float(probs[pred_id]),
        "probs": prob_dict,
    }


def format_topk(probs_dict, k=3):
    items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    return "\n".join([f"{lbl}: {p:.3f}" for lbl, p in items])


# =========================
# Sidebar config
# =========================

st.sidebar.header("⚙️ Cấu hình")

use_cuda_sidebar = st.sidebar.checkbox("Dùng GPU nếu có", value=True)
device_str = "cuda" if use_cuda_sidebar and torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Device đang dùng: `{device_str}`")

max_len = st.sidebar.slider("Max sequence length", 32, 256, 128, step=8)

pho_dir_default   = "phobert_vsmec_best_model"
xlmr_dir_default  = "xlmr_vsmec_best_model"
mbert_dir_default = "mbert_vsmec_best_model"

st.sidebar.markdown("### Thư mục model")
pho_dir = st.sidebar.text_input("PhoBERT", value=pho_dir_default)
xlmr_dir = st.sidebar.text_input("XLM-R", value=xlmr_dir_default)
mbert_dir = st.sidebar.text_input("mBERT", value=mbert_dir_default)

st.sidebar.markdown("### Chọn mô hình sử dụng")
use_phobert = st.sidebar.checkbox("Dùng PhoBERT", value=True)
use_xlmr    = st.sidebar.checkbox("Dùng XLM-R", value=False)
use_mbert   = st.sidebar.checkbox("Dùng mBERT", value=False)

if USE_UTS:
    st.sidebar.success("underthesea: đang dùng để tách từ cho PhoBERT.")
else:
    st.sidebar.warning("underthesea KHÔNG dùng được, PhoBERT sẽ tách từ đơn giản.")
    if uts_error:
        with st.sidebar.expander("Chi tiết lỗi underthesea"):
            st.code(uts_error)

# =========================
# Kiểm tra thư mục + load model được chọn
# =========================

bundles = {}

try:
    if use_phobert:
        if not os.path.isdir(pho_dir):
            st.error(f"Thư mục PhoBERT không tồn tại: `{pho_dir}`")
            st.stop()
        bundles["PhoBERT"] = load_model_bundle(pho_dir, device=device_str, use_fast=False)

    if use_xlmr:
        if not os.path.isdir(xlmr_dir):
            st.error(f"Thư mục XLM-R không tồn tại: `{xlmr_dir}`")
            st.stop()
        bundles["XLM-R"] = load_model_bundle(xlmr_dir, device=device_str, use_fast=True)

    if use_mbert:
        if not os.path.isdir(mbert_dir):
            st.error(f"Thư mục mBERT không tồn tại: `{mbert_dir}`")
            st.stop()
        bundles["mBERT"] = load_model_bundle(mbert_dir, device=device_str, use_fast=True)
except Exception as e:
    st.error(f"Không load được model.\nChi tiết lỗi:\n{e}")
    st.stop()

if not bundles:
    st.warning("Bạn chưa chọn mô hình nào ở sidebar. Hãy bật ít nhất 1 checkbox.")
    st.stop()
else:
    st.sidebar.success(f"Đã load {len(bundles)} mô hình: {', '.join(bundles.keys())}")


# =========================
# Giao diện nhập câu
# =========================

st.markdown("### Nhập câu cần dự đoán")

sample_text = "phấn chấn lên nào bro, ở đây có anh em, không phải lo"
input_text = st.text_area(
    "Mỗi dòng là một câu (nhiều dòng = dự đoán nhiều câu cùng lúc):",
    height=150,
    value=sample_text,
)

if st.button("🚀 Dự đoán"):
    if not input_text.strip():
        st.warning("Vui lòng nhập ít nhất một câu.")
    else:
        sentences = [line.strip() for line in input_text.split("\n") if line.strip()]

        st.markdown("## Kết quả")

        for idx, sent in enumerate(sentences, start=1):
            st.markdown("---")
            st.markdown(f"### Câu {idx}")
            st.markdown(f"**Câu gốc:** {sent}")

            with st.spinner("Đang dự đoán..."):
                # chuẩn bị columns tương ứng số model
                model_names = list(bundles.keys())
                cols = st.columns(len(model_names))

                for col_idx, name in enumerate(model_names):
                    bundle = bundles[name]
                    use_seg = (name == "PhoBERT")  # chỉ PhoBERT dùng underthesea
                    res = predict_one(
                        sent,
                        bundle,
                        max_length=max_len,
                        use_seg=use_seg,
                    )

                    with cols[col_idx]:
                        st.subheader(name)
                        st.markdown(f"**Nhãn:** `{res['pred_label']}`")
                        if name == "PhoBERT":
                            st.caption("Sau khi tách từ:")
                        else:
                            st.caption("Sau khi chuẩn hoá:")
                        st.code(res["processed"])
                        st.markdown("**Top-3 xác suất:**")
                        st.code(format_topk(res["probs"], k=3))

st.markdown("---")
st.caption("Demo phân loại cảm xúc VSMEC • PhoBERT / XLM-R / mBERT (fine-tuned)")
