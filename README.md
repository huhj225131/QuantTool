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

- 🚀 **Quantization & TensorRT Engine Build (Tab 3)**:
  - **Phần 1 - ONNX Model Quantization (NVIDIA ModelOpt)**:
    - Tối ưu hóa mô hình bằng **NVIDIA ModelOpt** hỗ trợ các chế độ `INT8`, `FP8`, `INT4`.
    - **Real-time Log Streaming**: Hiển thị trực tiếp log thực thi của quá trình quantize theo thời gian thực.
    - **Tự động ánh xạ tên Node (Graph Input Mapping)**: Tự đọc đồ thị ONNX để map khớp tên input tensor với mảng calibration data.
    - Tự động điền mô hình ONNX đã quantize sang Phần 2 bên dưới.
  - **Phần 2 - TensorRT Engine Build (`trtexec`)**:
    - Biên dịch mô hình ONNX (đặc biệt là mô hình ONNX đã quantize) sang định dạng **TensorRT Engine (`.engine`)** bằng công cụ `trtexec`.
    - Hỗ trợ cờ `--stronglyTyped` (bắt buộc giữ nguyên kiểu dữ liệu quantize), `--fp16`, `--int8` và các cờ CLI bổ sung tùy chỉnh.
    - **Real-time trtexec Execution Log**: Stream toàn bộ log tiến trình biên dịch engine từng dòng ngay trên màn hình.

- 💡 **Smart Path Autocomplete**: Gõ đường dẫn trực tiếp trên ô nhập với gợi ý thả xuống (dropdown) ưu tiên các thư mục có chứa file mục tiêu (`.onnx`, `.npy`, `.engine`, ảnh thô).

---

## 📋 Yêu cầu hệ thống

- **OS**: Linux (Ubuntu 20.04/22.04+ được khuyến nghị)
- **Python**: 3.10+
- **Hardware**: GPU NVIDIA (khuyến nghị có CUDA/cuDNN và TensorRT cho tốc độ quantize và build engine tối ưu).

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

## 🐳 Đóng gói & Chạy bằng Docker (Nhánh `jetson` - L4T r36.4.0)

Nhánh **`jetson`** được cấu hình sẵn base image **`dustynv/l4t-ml:r36.4.0`** (JetPack 6 / L4T r36.4.0) tối ưu cho các thiết bị **NVIDIA Jetson** (Orin Nano, Orin NX, AGX Orin) tích hợp sẵn CUDA, cuDNN, PyTorch và công cụ `trtexec`.

### Cách 1: Sử dụng Docker Compose trên Jetson (Khuyên dùng)

```bash
# Build và chạy ứng dụng trên thiết bị Jetson
docker compose up -d --build

# Xem log thực thi
docker logs -f quant_tool_jetson

# Dừng container
docker compose down
```

### Cách 2: Sử dụng Docker CLI thủ công trên Jetson

```bash
# 1. Build image từ dustynv/l4t-ml:r36.4.0
docker build -t quant_tool:jetson-l4t-r36.4.0 .

# 2. Chạy container trên Jetson với runtime nvidia
docker run -d --runtime nvidia \
  -p 7860:7860 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  --name quant_tool_jetson \
  quant_tool:jetson-l4t-r36.4.0
```

Mở trình duyệt truy cập: `http://localhost:7860`.

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
   - Chọn thư mục chứa ảnh thô, chỉnh các thông số Preprocessing → Nhấn **Tạo Dataset (.npy)**.
   - Chuyển sang **Accordion 2 (Validator)** → Nhấn **Validate Data** để đảm bảo file `.npy` tương thích hoàn toàn với model.
3. **Bước 3 (Quantize ONNX & Build TensorRT Engine)**: Vào **Tab 3 (Quantization & Engine Build)**:
   - **Phần 1**: Nhấn **Start ONNX Quantization** để quantize mô hình bằng `nvidia-modelopt`.
   - **Phần 2**: Nhấn **Build TensorRT Engine** để biên dịch mô hình ONNX sang tệp `.engine` bằng `trtexec`. Quan sát log thực thi thời gian thực ở cả 2 bước.

---

## 📁 Cấu trúc dự án

```
quant_tool/
├── app.py                    # Entry point chính của ứng dụng Gradio
├── src/
│   ├── onnx_inspector.py     # Phân tích metadata & yêu cầu input mô hình ONNX
│   ├── data_generator.py     # Tạo dataset calibration (.npy) qua OpenCV hoặc Python Script
│   ├── data_validator.py     # Validate file .npy/.npz so với ONNX input
│   ├── quantizer.py          # Wrapper cho NVIDIA ModelOpt ONNX Quantization & streaming log
│   ├── engine_builder.py     # Wrapper biên dịch TensorRT Engine bằng trtexec & streaming log
│   └── utils.py              # Tiện ích autocomplete đường dẫn thông minh, logging
├── ui/
│   └── gradio_app.py         # Thiết kế giao diện người dùng Gradio 3 Tabs (Orange theme)
├── configs/
│   └── default_config.yaml   # Cấu hình tham số mặc định
├── outputs/                  # Thư mục lưu mô hình .onnx/.engine đã quantize (.gitignored)
├── requirements.txt          # Danh sách thư viện Python cần thiết
├── setup.py                  # Setup package
├── Dockerfile                # File đóng gói Docker Image với PyTorch & CUDA 12.1
├── docker-compose.yml        # Docker Compose configuration cho GPU runtime
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 📝 License

Dự án được phát hành theo giấy phép MIT.

