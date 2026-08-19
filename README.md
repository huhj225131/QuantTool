# QuantTool – ONNX Model Optimization Tool

> Tool tối ưu mô hình ONNX sử dụng [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) với giao diện [Gradio](https://gradio.app/).

## ✨ Tính năng

- 📦 **Model Inspector**: Upload file ONNX → tự động phân tích input/output shapes, dtypes, opset version
- 📊 **Calibration Data Validator**: Validate dữ liệu calibration (.npy/.npz) so với yêu cầu model
- 🚀 **Quantization**: Quantize model ONNX bằng NVIDIA ModelOpt (INT8, FP8, INT4)

## 📋 Yêu cầu

- Python 3.10+
- Conda (khuyến nghị)

## 🔧 Cài đặt

```bash
# 1. Activate conda environment
conda activate quant_tool

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. (Optional) Cài nvidia-modelopt nếu cần quantize
pip install nvidia-modelopt[onnx]
```

## 🚀 Sử dụng

```bash
# Chạy Gradio UI
python app.py

# Chạy với cổng tùy chỉnh
python app.py --port 8080

# Tạo public link
python app.py --share
```

Mở trình duyệt tại `http://localhost:7860`.

## 📁 Cấu trúc project

```
quant_tool/
├── app.py                    # Entry point
├── src/
│   ├── onnx_inspector.py     # Đọc ONNX metadata
│   ├── data_validator.py     # Validate calibration data
│   ├── quantizer.py          # Wrapper ModelOpt quantization
│   └── utils.py              # Tiện ích chung
├── ui/
│   └── gradio_app.py         # Gradio UI definition
├── configs/
│   └── default_config.yaml   # Cấu hình mặc định
├── outputs/                  # Model đã quantize (gitignored)
├── tests/                    # Unit tests
├── requirements.txt
├── setup.py
├── CHANGELOG.md
└── README.md
```

## 📝 License

MIT
