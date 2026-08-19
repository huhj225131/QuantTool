# Changelog

Tất cả các thay đổi quan trọng của project sẽ được ghi lại tại đây.

Format dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-19

### Added
- Khởi tạo project QuantTool
- Module `onnx_inspector`: Đọc và phân tích metadata từ file ONNX (input/output shapes, dtypes, opset version)
- Module `data_validator`: Validate calibration data (.npy/.npz) so với model input requirements
- Module `quantizer`: Wrapper cho `modelopt.onnx.quantization.quantize()` hỗ trợ INT8, FP8, INT4
- Gradio UI với 3 tabs: Model Inspector, Calibration Data, Quantization
- Cấu hình mặc định (`configs/default_config.yaml`)
- Project structure chuẩn với git, requirements.txt, README, CHANGELOG
