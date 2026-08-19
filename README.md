# QuantTool – ONNX Model Optimization Tool 🚀

> Công cụ giao diện trực quan ([Gradio](https://gradio.app/)) hỗ trợ phân tích, chuẩn bị dữ liệu hiệu chỉnh (calibration) và tối ưu hóa/lượng tử hóa (quantization) các mô hình AI định dạng **ONNX** dựa trên **[NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** (`nvidia-modelopt`).

---

## ✨ Tính năng nổi bật

- 📦 **Model Inspector (Tab 1)**: 
  - Upload / chọn mô hình ONNX với gợi ý đường dẫn thông minh.
  - Phân tích chi tiết metadata: Opset version, IR version, tổng số node, danh sách Input / Output (name, shape, dtype).
  - Tự động suy luận **Yêu cầu Calibration Data** (VD: `[N, 3, 640, 640]`, `float32`).
  - Tự động đồng bộ đường dẫn ONNX & kích thước (Width, Height) sang các Tab khác.

- 🛠️ **Calibration Data (Tab 2)**:
  - **Generator (Tạo dữ liệu .npy từ ảnh thô)**:
    - **OpenCV Pipeline**: Hỗ trợ chế độ Resize (*Letterbox / Top-Left Padding*, *Direct Resize*, *Center Crop*), chuẩn hóa (*[0, 1]*, *[-1, 1]*, *ImageNet Z-score*), chuyển đổi kênh màu (*BGR*, *RGB*) và kiểu dữ liệu (*float32*, *float16*). Quét thư mục ảnh đệ quy với cơ chế dừng sớm (Early Exit) tăng tốc tối đa.
    - **Custom Python Script**: Tự do viết code tiền xử lý tùy chỉnh trực tiếp trên Code Editor hoặc nạp từ file `.py`.
  - **Validator (Kiểm tra dữ liệu .npy / .npz)**: Đối chiếu shape, dtype, kiểm tra giá trị NaN/Inf và dải min/max/mean/std so với mô hình ONNX.
  - Tự động điền file `.npy` vừa tạo sang Tab Quantization.

- ⚡ **Model Quantization (Tab 3)**:
  - Tối ưu hóa mô hình bằng **NVIDIA ModelOpt** hỗ trợ các chế độ:
    - **INT8**: Calibration method `max`, `entropy`, `minmax`.
    - **FP8**: Dành cho GPU kiến trúc NVIDIA Ada Lovelace / Hopper.
    - **INT4**: Thuật toán `awq_clip`, `rtn_dq` phù hợp băng thông bộ nhớ thấp (low-batch).
  - **Tự động ánh xạ tên Node (Graph Input Mapping)**: Tự đọc đồ thị ONNX để map khớp tên input tensor (`input.1`, `images`,...) với mảng calibration data, loại bỏ lỗi lệch key.
  - Hỗ trợ xuất mô hình lớn > 2GB (External Data format).

- 💡 **Smart Path Autocomplete**: Gõ đường dẫn trực tiếp trên ô nhập với gợi ý thả xuống (dropdown) ưu tiên các thư mục có chứa file mục tiêu (`.onnx`, `.npy`, ảnh thô).

---

## 📋 Yêu cầu hệ thống

- **OS**: Linux (Ubuntu 20.04/22.04+ được khuyến nghị)
- **Python**: 3.10+
- **Conda** (khuyến nghị)
- **Hardware**: GPU NVIDIA (khuyến nghị có CUDA/cuDNN cho tốc độ quantize tối ưu; hỗ trợ chạy CPU cho mô hình nhỏ).

---

## 🔧 Cài đặt

```bash
# 1. Kích hoạt môi trường conda
conda activate quant_tool

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. (Tùy chọn) Cài đặt nvidia-modelopt nếu chưa có
pip install nvidia-modelopt[onnx]
```

---

## 🚀 Hướng dẫn sử dụng

Chạy ứng dụng Gradio:

```bash
# Chạy ứng dụng (mặc định tại http://localhost:7860)
python app.py

# Chạy với cổng tùy chỉnh
python app.py --port 8080

# Tạo liên kết chia sẻ công khai (Public Share Link)
python app.py --share
```

Mở trình duyệt truy cập: `http://localhost:7860`.

---

## 📖 Quy trình làm việc đề xuất (Workflow)

1. **Bước 1 (Inspect)**: Vào **Tab 1 (Model Inspector)** → Nhập/chọn file `.onnx` → Nhấn **Inspect Model** để xem cấu trúc và thông số Input (`Width`, `Height`, `Channel`, `Dtype`).
2. **Bước 2 (Generate & Validate Calib Data)**: 
   - Vào **Tab 2 (Calibration Data)** → Mở **Accordion 1 (Generator)**.
   - Chọn thư mục chứa ảnh thô, chỉnh các thông số Preprocessing (chọn Letterbox nếu là mô hình YOLO/Detection) → Nhấn **Tạo Dataset (.npy)**.
   - Chuyển sang **Accordion 2 (Validator)** → Nhấn **Validate Data** để đảm bảo file `.npy` tương thích hoàn toàn với model.
3. **Bước 3 (Quantize)**: Vào **Tab 3 (Quantization)** → Chọn `quantize_mode` (mặc định INT8) và `calibration_method` (`max` hoặc `entropy`) → Nhấn **Bắt đầu Quantize**. Mô hình xuất ra được lưu tại thư mục `outputs/` hoặc đường dẫn tùy chọn.

---

## 📁 Cấu trúc dự án

```
quant_tool/
├── app.py                    # Entry point chính của ứng dụng Gradio
├── src/
│   ├── onnx_inspector.py     # Phân tích metadata & yêu cầu input mô hình ONNX
│   ├── data_generator.py     # Tạo dataset calibration (.npy) qua OpenCV hoặc Python Script
│   ├── data_validator.py     # Validate file .npy/.npz so với ONNX input
│   ├── quantizer.py          # Wrapper chính cho NVIDIA ModelOpt Quantization
│   └── utils.py              # Tiện ích autocomplete đường dẫn thông minh, logging
├── ui/
│   └── gradio_app.py         # Thiết kế giao diện người dùng (Gradio UI - Orange theme)
├── configs/
│   └── default_config.yaml   # Cấu hình tham số mặc định
├── outputs/                  # Thư mục lưu mô hình đã quantize (.gitignored)
├── requirements.txt          # Danh sách thư viện Python cần thiết
├── setup.py                  # Setup package
├── CHANGELOG.md              # Nhật ký thay đổi qua từng phiên bản
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 📝 License

Dự án được phát hành theo giấy phép MIT.

