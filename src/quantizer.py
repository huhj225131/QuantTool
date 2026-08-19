"""
quantizer.py – Wrapper cho NVIDIA ModelOpt ONNX Quantization

Cung cấp interface thống nhất để gọi modelopt.onnx.quantization.quantize()
với các tham số được cấu hình từ UI.

Hỗ trợ:
- Quantize modes: int8, fp8, int4
- Calibration methods: minmax, entropy, max
- Chạy được trên cả CPU và GPU (CPU sẽ chậm hơn với model lớn)
"""

import os
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("quant_tool.quantizer")


@dataclass
class QuantConfig:
    """Cấu hình cho quá trình quantization.

    Attributes:
        onnx_path: Đường dẫn model ONNX gốc.
        output_path: Đường dẫn xuất model đã quantize.
        quantize_mode: Chế độ quantize (int8, fp8, int4).
        calibration_method: Thuật toán calibration (minmax, entropy, max).
        calibration_data_path: Đường dẫn file .npy/.npz chứa calibration data.
        op_types_to_quantize: Chỉ quantize các op types này (None = tất cả).
        nodes_to_exclude: Danh sách node names cần loại trừ khỏi quantization.
        use_external_data_format: Lưu weights riêng cho model > 2GB.
    """

    onnx_path: str = ""
    output_path: str = "outputs/model_quantized.onnx"
    quantize_mode: str = "int8"
    calibration_method: str = "minmax"
    calibration_data_path: str = ""
    op_types_to_quantize: Optional[list[str]] = None
    nodes_to_exclude: list[str] = field(default_factory=list)
    use_external_data_format: bool = False


def _load_calibration_data(data_path: str) -> dict[str, np.ndarray]:
    """Load calibration data từ file .npy hoặc .npz.

    Args:
        data_path: Đường dẫn tới file calibration data.

    Returns:
        Dict mapping tên input → numpy array.

    Raises:
        ValueError: Nếu format không hỗ trợ.
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npy":
        data = np.load(data_path)
        # Trả về với key mặc định, quantize() sẽ map theo thứ tự
        return {"input": data}
    elif ext == ".npz":
        npz = np.load(data_path)
        return dict(npz)
    else:
        raise ValueError(f"Format calibration data không hỗ trợ: {ext}")


def check_modelopt_available() -> tuple[bool, str]:
    """Kiểm tra xem nvidia-modelopt có được cài đặt hay không.

    Returns:
        Tuple (is_available, message).
    """
    try:
        import modelopt.onnx.quantization as moq  # noqa: F401
        return True, "nvidia-modelopt[onnx] đã sẵn sàng ✅"
    except ImportError as e:
        return False, (
            f"nvidia-modelopt[onnx] chưa được cài đặt ❌\n"
            f"Lỗi: {e}\n"
            f"Cài đặt bằng: pip install nvidia-modelopt[onnx]"
        )


def quantize_model(
    config: QuantConfig,
    progress_callback=None,
) -> dict[str, Any]:
    """Thực hiện quantization model ONNX bằng NVIDIA ModelOpt.

    Args:
        config: QuantConfig chứa toàn bộ tham số.
        progress_callback: Optional callable(progress_float, message_str)
                          để cập nhật progress trên UI.

    Returns:
        Dict chứa success (bool), message (str), output_path, duration_seconds.
    """
    def _log(msg, progress=None):
        logger.info(msg)
        if progress_callback and progress is not None:
            progress_callback(progress, msg)

    # Validate inputs
    if not os.path.exists(config.onnx_path):
        return {
            "success": False,
            "message": f"File ONNX không tồn tại: {config.onnx_path}",
        }

    if not os.path.exists(config.calibration_data_path):
        return {
            "success": False,
            "message": f"Calibration data không tồn tại: {config.calibration_data_path}",
        }

    # Check modelopt
    available, avail_msg = check_modelopt_available()
    if not available:
        return {"success": False, "message": avail_msg}

    # Import modelopt (đã kiểm tra ở trên)
    import modelopt.onnx.quantization as moq

    # Tạo output directory nếu chưa có
    output_dir = os.path.dirname(config.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    _log("🔄 Đang load calibration data...", 0.1)

    try:
        calib_data = _load_calibration_data(config.calibration_data_path)
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi khi load calibration data: {e}",
        }

    _log(f"📊 Calibration data loaded: {list(calib_data.keys())}", 0.2)
    _log(f"⚙️ Bắt đầu quantize (mode={config.quantize_mode}, method={config.calibration_method})...", 0.3)

    start_time = time.time()

    try:
        # Gọi ModelOpt quantize
        # API: modelopt.onnx.quantization.quantize(
        #     onnx_path, calibration_data=..., quantize_mode=...,
        #     calibration_method=..., output_path=..., ...
        # )
        quantize_kwargs = {
            "onnx_path": config.onnx_path,
            "quantize_mode": config.quantize_mode,
            "output_path": config.output_path,
        }

        # Calibration data – thử truyền trực tiếp
        quantize_kwargs["calibration_data"] = calib_data

        # Calibration method
        if config.calibration_method:
            quantize_kwargs["calibration_method"] = config.calibration_method

        # Op types to quantize
        if config.op_types_to_quantize:
            quantize_kwargs["op_types_to_quantize"] = config.op_types_to_quantize

        _log("🚀 Đang chạy quantization...", 0.5)

        moq.quantize(**quantize_kwargs)

        duration = time.time() - start_time

        _log(f"✅ Quantization hoàn thành trong {duration:.1f}s!", 1.0)

        # Kiểm tra output file
        if os.path.exists(config.output_path):
            output_size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
            input_size_mb = os.path.getsize(config.onnx_path) / (1024 * 1024)
            compression = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0

            return {
                "success": True,
                "message": (
                    f"✅ Quantization thành công!\n"
                    f"- Output: {config.output_path}\n"
                    f"- Kích thước gốc: {input_size_mb:.2f} MB\n"
                    f"- Kích thước sau quantize: {output_size_mb:.2f} MB\n"
                    f"- Giảm: {compression:.1f}%\n"
                    f"- Thời gian: {duration:.1f}s"
                ),
                "output_path": config.output_path,
                "duration_seconds": duration,
                "original_size_mb": input_size_mb,
                "quantized_size_mb": output_size_mb,
                "compression_percent": compression,
            }
        else:
            return {
                "success": False,
                "message": "Quantization hoàn thành nhưng không tìm thấy file output",
            }

    except Exception as e:
        duration = time.time() - start_time
        logger.exception("Quantization failed")
        return {
            "success": False,
            "message": f"❌ Lỗi trong quá trình quantization:\n{type(e).__name__}: {e}",
            "duration_seconds": duration,
        }
