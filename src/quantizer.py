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
from typing import Any, Optional, Generator
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
    calibration_method: str = "max"
    calibration_data_path: str = ""
    op_types_to_quantize: Optional[list[str]] = None
    nodes_to_exclude: list[str] = field(default_factory=list)
    use_external_data_format: bool = False


def _load_calibration_data(data_path: str, onnx_path: Optional[str] = None) -> dict[str, np.ndarray]:
    """Load calibration data từ file .npy hoặc .npz và khớp key với input name của ONNX model.

    Args:
        data_path: Đường dẫn tới file calibration data.
        onnx_path: Đường dẫn tới file ONNX model (để đọc chính xác tên input node).

    Returns:
        Dict mapping tên input node chính xác → numpy array.

    Raises:
        ValueError: Nếu format không hỗ trợ hoặc file bị lỗi.
    """
    ext = os.path.splitext(data_path)[1].lower()

    # Trích xuất danh sách tên input node thực tế từ graph ONNX
    input_names = []
    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnx
            model = onnx.load(onnx_path)
            initializer_names = {init.name for init in model.graph.initializer}
            input_names = [inp.name for inp in model.graph.input if inp.name not in initializer_names]
        except Exception as e:
            logger.warning(f"Không thể đọc input names từ ONNX model: {e}")

    default_input_name = input_names[0] if input_names else "input"

    if ext == ".npy":
        data = np.load(data_path)
        logger.info(f"Loaded .npy array shape {data.shape}, dtype {data.dtype}, mapping to ONNX input name '{default_input_name}'")
        return {default_input_name: data}

    elif ext == ".npz":
        npz = np.load(data_path)
        raw_dict = dict(npz)

        # Nếu npz chỉ có 1 key và khác với ONNX input_name, re-key lại cho khớp
        if len(raw_dict) == 1 and input_names:
            key = list(raw_dict.keys())[0]
            if key not in input_names:
                logger.info(f"Re-keying npz array '{key}' -> '{default_input_name}'")
                return {default_input_name: raw_dict[key]}

        return raw_dict
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


class QuantLogStreamer:
    """Helper lưu vết log và stream cho Gradio UI."""

    def __init__(self):
        self.logs: list[str] = []

    def log(self, msg: str) -> str:
        self.logs.append(msg)
        return "\n".join(self.logs)

    def get_full_log(self) -> str:
        return "\n".join(self.logs)


def quantize_model_stream(
    config: QuantConfig,
) -> Generator[tuple[str, str], None, dict]:
    """Generator thực hiện quantization model ONNX bằng NVIDIA ModelOpt và stream log liên tục.

    Args:
        config: QuantConfig chứa toàn bộ tham số.

    Yields:
        Tuple (current_logs: str, status_summary: str)
    """
    streamer = QuantLogStreamer()

    def _append(msg: str) -> str:
        logger.info(msg)
        return streamer.log(msg)

    # Attach custom logging handler để bắt log từ modelopt package
    handler = logging.StreamHandler()
    class _CustomStream:
        def write(self, text):
            if text and text.strip():
                streamer.log(text.strip())
        def flush(self):
            pass

    # Attach handler to root & modelopt logger
    custom_handler = logging.StreamHandler(_CustomStream())
    custom_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    modelopt_logger = logging.getLogger("modelopt")
    modelopt_logger.addHandler(custom_handler)

    try:
        # Validate inputs
        if not os.path.exists(config.onnx_path):
            err_msg = f"⚠️ File ONNX không tồn tại: `{config.onnx_path}`"
            curr_log = _append(err_msg)
            yield curr_log, err_msg
            return {"success": False, "message": err_msg}

        if not os.path.exists(config.calibration_data_path):
            err_msg = f"⚠️ File calibration data không tồn tại: `{config.calibration_data_path}`"
            curr_log = _append(err_msg)
            yield curr_log, err_msg
            return {"success": False, "message": err_msg}

        # Check modelopt
        available, avail_msg = check_modelopt_available()
        if not available:
            curr_log = _append(avail_msg)
            yield curr_log, avail_msg
            return {"success": False, "message": avail_msg}

        import modelopt.onnx.quantization as moq

        output_dir = os.path.dirname(config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        curr_log = _append("🔄 Đang load và ánh xạ calibration data với ONNX graph input names...")
        yield curr_log, "⏳ Đang chuẩn bị dữ liệu calibration..."

        try:
            calib_data = _load_calibration_data(config.calibration_data_path, onnx_path=config.onnx_path)
        except Exception as e:
            err_msg = f"❌ Lỗi khi load calibration data: {e}"
            curr_log = _append(err_msg)
            yield curr_log, err_msg
            return {"success": False, "message": err_msg}

        curr_log = _append(f"📊 Calibration data loaded & mapped keys: {list(calib_data.keys())}")
        curr_log = _append(f"⚙️ Khởi tạo ModelOpt Quantizer (mode={config.quantize_mode}, method={config.calibration_method})...")
        yield curr_log, "⏳ Đang thực thi NVIDIA ModelOpt Quantization..."

        start_time = time.time()

        quantize_kwargs = {
            "onnx_path": config.onnx_path,
            "quantize_mode": config.quantize_mode,
            "calibration_data": calib_data,
            "calibration_method": config.calibration_method,
            "output_path": config.output_path,
        }

        if config.op_types_to_quantize:
            quantize_kwargs["op_types_to_quantize"] = config.op_types_to_quantize

        curr_log = _append(
            f"🚀 Đang chạy moq.quantize("
            f"onnx_path='{config.onnx_path}', "
            f"quantize_mode='{config.quantize_mode}', "
            f"calibration_data={list(calib_data.keys())}, "
            f"calibration_method='{config.calibration_method}', "
            f"output_path='{config.output_path}')"
        )
        yield curr_log, "⏳ NVIDIA ModelOpt đang tính toán calibration & quantize nodes..."

        moq.quantize(**quantize_kwargs)

        duration = time.time() - start_time

        curr_log = _append(f"✅ Quantization hoàn thành trong {duration:.1f}s!")

        if os.path.exists(config.output_path):
            output_size_mb = os.path.getsize(config.output_path) / (1024 * 1024)
            input_size_mb = os.path.getsize(config.onnx_path) / (1024 * 1024)
            compression = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0

            success_msg = (
                f"✅ **Quantization thành công!**\n"
                f"- **Output**: `{config.output_path}`\n"
                f"- **Kích thước gốc**: `{input_size_mb:.2f} MB`\n"
                f"- **Kích thước sau quantize**: `{output_size_mb:.2f} MB` (giảm {compression:.1f}%)\n"
                f"- **Thời gian thực thi**: `{duration:.1f}s`"
            )
            curr_log = _append(f"\n🎉 SUCCESS: Quantized ONNX saved to {config.output_path} ({output_size_mb:.2f} MB)")
            yield curr_log, success_msg
            return {
                "success": True,
                "message": success_msg,
                "output_path": config.output_path,
                "duration_seconds": duration,
            }
        else:
            fail_msg = "❌ Quantization hoàn thành nhưng không tìm thấy file output"
            curr_log = _append(fail_msg)
            yield curr_log, fail_msg
            return {"success": False, "message": fail_msg}

    except Exception as e:
        duration = time.time() - start_time
        err_msg = f"❌ Lỗi trong quá trình quantization:\n{type(e).__name__}: {e}"
        logger.exception("Quantization failed")
        curr_log = _append(f"\n💥 EXCEPTION: {err_msg}")
        yield curr_log, err_msg
        return {"success": False, "message": err_msg}

    finally:
        modelopt_logger.removeHandler(custom_handler)


def quantize_model(
    config: QuantConfig,
    progress_callback=None,
) -> dict[str, Any]:
    """Wrapper tương thích ngược cho quantize_model (không stream)."""
    last_log = ""
    last_status = ""
    for curr_log, curr_status in quantize_model_stream(config):
        last_log = curr_log
        last_status = curr_status
    return {"success": "✅" in last_status, "message": last_status, "log": last_log}

