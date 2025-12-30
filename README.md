# Vietnamese Sentiment Analysis (Phân tích 7 cảm xúc trong văn bản tiếng Việt)

## 📖 Giới thiệu (Overview)
Dự án này tập trung vào việc nghiên cứu và so sánh hiệu năng của các mô hình ngôn ngữ lớn (Transformer-based models) để giải quyết bài toán **Phân tích cảm xúc văn bản tiếng Việt** (Sentiment Analysis).

Mục tiêu là xây dựng một hệ thống có khả năng phân loại cảm xúc từ các bình luận mạng xã hội hoặc đánh giá sản phẩm, đồng thời triển khai một ứng dụng web demo đơn giản để kiểm thử thực tế.

## 🗂 Cấu trúc Dự án (Project Structure)
Dự án bao gồm các thành phần chính:

* **Notebooks (Thực nghiệm mô hình):**
    * `baseline.ipynb`: Các mô hình máy học cơ bản làm cơ sở so sánh.
    * `mbert.ipynb` & `mbert-no-other.ipynb`: Huấn luyện và tinh chỉnh mô hình **mBERT**.
    * `xlmr.ipynb` & `xlmr-no-other.ipynb`: Huấn luyện và tinh chỉnh mô hình **XLM-Roberta**.
    * `phobert.ipynb` & `phobert-no-other.ipynb`: Huấn luyện và tinh chỉnh mô hình **PhoBERT**.
* **Application:**
    * `app.py`: Mã nguồn ứng dụng web (xây dựng bằng Streamlit/Flask) để demo dự đoán cảm xúc.
* **Dataset:**
    * Dữ liệu được chia theo tỷ lệ 8:1:1 (Train: 80%, Test: 10%, Valid: 10%).
    * `train_nor_811.xlsx`, `valid_nor_811.xlsx`, `test_nor_811.xlsx`.

## 🚀 Cài đặt và Chạy thử (Installation & Usage)

### 1. Yêu cầu môi trường
* Python 3.8+
* Các thư viện chính: `transformers`, `torch`, `pandas`, `scikit-learn`, `streamlit` (nếu dùng cho app.py).

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
