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
- Integrated `gr.Dropdown(allow_custom_value=True)` cho phép gõ đường dẫn và hiển thị gợi ý thả xuống trực tiếp ngay tại ô nhập (không cần ô gợi ý riêng biệt ở dưới)
- Mặc định gợi ý đường dẫn bắt đầu từ thư mục cha (`os.path.dirname(os.getcwd())`), giúp liệt kê ngay các file/folder ngang hàng với `quant_tool`
- Nâng cấp **Hệ thống Gợi ý Thư mục Thông minh (Smart Directory Prioritization)**: Tự động phân tích nội dung thư mục con và ưu tiên đẩy các thư mục **CÓ CHỨA FILE MỤC TIÊU** (`.onnx`, `.npy`, ảnh thô...) lên đầu danh sách gợi ý.
- Thêm ô chọn đường dẫn ONNX Model cho Tab 2 (Calibration Data) và Tab 3 (Quantization)
- Tự động đồng bộ đường dẫn ONNX Model và Calibration Data giữa các Tab khi Inspect hoặc Validate thành công
- Cập nhật danh sách thuật toán calibration động theo `quantize_mode` (INT8/FP8: `max`, `entropy`; INT4: `awq_clip`, `rtn_dq`)
- Tích hợp hướng dẫn và ghi chú phần cứng (Hopper/Ada cho FP8, Low-batch memory bandwidth cho INT4 AWQ) trực tiếp trên giao diện
- Nâng cấp **Quét ảnh đệ quy với cơ chế dừng sớm (Early Exit)**: Quét đệ quy toàn bộ thư mục con và dừng ngay khi đạt đủ `max_samples` ảnh yêu cầu, giúp tăng tốc tối đa khi làm việc với thư mục dung lượng lớn.
- Sửa triệt để lỗi Quantization (`AssertionError` / `ValueError`): Tự động đọc và ánh xạ tên input node chính xác từ graph ONNX (`"input.1"`, `"images"`,...) sang dict calibration data trước khi gọi `moq.quantize()`.
- Chuyển giao diện Gradio về tông **Màu Cam Mặc Định (`orange` theme)** theo sở thích người dùng.
- Chuẩn hóa hàm gọi `modelopt.onnx.quantization.quantize()` trong [src/quantizer.py](file:///media/hung/DATA/jeson/quant_tool/src/quantizer.py) khớp chính xác 100% với công thức API chính thức của NVIDIA ModelOpt: `quantize(onnx_path=..., quantize_mode=..., calibration_data=..., calibration_method=..., output_path=...)`
- Tích hợp bộ **Calibration Data Generator** trong Tab 2 với 2 phương án:
  - **OpenCV Pipeline (Chuẩn)**: Hỗ trợ `Letterbox / Top-Left Padding`, `Direct Resize`, `Center Crop`, `cv2.dnn.blobFromImage`, chuyển đổi màu, chuẩn hóa `[0,1]`, `[-1,1]`, `ImageNet Z-score` và xuất `float32`/`float16`.
  - **Custom Python Script**: Cho phép chọn file `.py` trực tiếp từ đĩa (với gợi ý autocomplete) HOẶC viết/chỉnh sửa code trực tiếp trong Code Editor. Có nút nạp code từ file `.py` vào editor.
- Tự động điền file `.npy` vừa tạo sang Validator (Tab 2) và Quantizer (Tab 3).
- Tự động điền kích thước `Width` & `Height` vào Generator khi Inspect mô hình ở Tab 1.
- Sửa lỗi hiển thị lặp lại dòng tiêu đề "Yêu cầu Calibration Data" trên giao diện Model Inspector
- Hiển thị chiều đầu tiên (Dim 0) trong yêu cầu Calibration Data là `N` (số lượng mẫu calibration, ví dụ `[N, 3, 640, 640]`) thay vì cố định theo batch size đơn lẻ
- Cấu hình mặc định (`configs/default_config.yaml`)
- Project structure chuẩn với git, requirements.txt, README, CHANGELOG
