"""
engine_builder.py – Wrapper thực thi trtexec để build TensorRT Engine (.engine) từ ONNX model.

Cung cấp interface streaming log thời gian thực qua subprocess.Popen.
"""

import os
import shutil
import subprocess
import time
import logging
from typing import Generator, Optional, Tuple, List

logger = logging.getLogger("quant_tool.engine_builder")

DEFAULT_TRTEXEC_PATHS = [
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/bin/trtexec",
    "/usr/bin/trtexec",
]


def find_trtexec_path(user_custom_path: Optional[str] = None) -> Tuple[bool, str]:
    """Tìm kiếm đường dẫn tới file trtexec binary.

    Args:
        user_custom_path: Đường dẫn do người dùng chỉ định (nếu có).

    Returns:
        Tuple (found: bool, path_or_err_msg: str)
    """
    if user_custom_path and user_custom_path.strip():
        custom_path = user_custom_path.strip()
        if os.path.exists(custom_path) and os.access(custom_path, os.X_OK):
            return True, custom_path
        elif os.path.exists(custom_path):
            return True, custom_path  # Vẫn thử chạy
        else:
            logger.warning(f"File trtexec custom không tồn tại: {custom_path}")

    # Kiểm tra trong các vị trí mặc định
    for path in DEFAULT_TRTEXEC_PATHS:
        if os.path.exists(path):
            return True, path

    # Kiểm tra trong PATH hệ thống
    which_path = shutil.which("trtexec")
    if which_path:
        return True, which_path

    return False, (
        "❌ Không tìm thấy công cụ `trtexec`!\n"
        "Vui lòng cài đặt TensorRT hoặc chỉ định chính xác đường dẫn `trtexec` (mặc định: /usr/src/tensorrt/bin/trtexec)."
    )


def build_engine_trtexec_stream(
    onnx_path: str,
    output_engine_path: str = "outputs/model_quantized.engine",
    extra_args: str = "",
    trtexec_path: Optional[str] = None,
) -> Generator[Tuple[str, str], None, dict]:
    """Build TensorRT engine bằng trtexec và yield log thời gian thực.

    Lệnh mặc định: trtexec --onnx=<onnx_path> --saveEngine=<output_engine_path> --stronglyTyped

    Args:
        onnx_path: Đường dẫn file ONNX nguồn.
        output_engine_path: Đường dẫn lưu file .engine xuất ra.
        extra_args: Cờ bổ sung tùy chỉnh (ví dụ: --workspace=4096).
        trtexec_path: Đường dẫn tới trtexec binary.

    Yields:
        Tuple (current_logs: str, status_summary: str)
    """
    logs: List[str] = []

    def _append_log(msg: str) -> str:
        logs.append(msg)
        return "\n".join(logs)

    onnx_path = onnx_path.strip()
    output_engine_path = output_engine_path.strip()

    if not os.path.exists(onnx_path):
        err_msg = f"❌ File ONNX không tồn tại: {onnx_path}"
        yield _append_log(err_msg), err_msg
        return {"success": False, "message": err_msg}

    # Tìm trtexec binary
    found, bin_path = find_trtexec_path(trtexec_path)
    if not found:
        yield _append_log(bin_path), bin_path
        return {"success": False, "message": bin_path}

    # Tạo thư mục output nếu chưa có
    out_dir = os.path.dirname(output_engine_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Xây dựng lệnh trtexec chuẩn
    cmd = [
        bin_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine_path}",
        "--stronglyTyped",
    ]

    if extra_args and extra_args.strip():
        # Phân tách các tham số bổ sung
        import shlex
        try:
            extra_tokens = shlex.split(extra_args.strip())
            cmd.extend(extra_tokens)
        except Exception as e:
            logger.warning(f"Lỗi parse extra_args '{extra_args}': {e}")
            cmd.extend(extra_args.strip().split())

    cmd_str = " ".join(cmd)
    logger.info(f"Khởi chạy trtexec: {cmd_str}")

    init_msg = f"🚀 Bắt đầu build TensorRT Engine...\n📌 Command: `{cmd_str}`\n"
    current_log = _append_log(init_msg)
    yield current_log, "⏳ Đang khởi chạy trtexec..."

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Đọc log theo từng dòng real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                line_str = line.rstrip()
                current_log = _append_log(line_str)
                yield current_log, "⏳ Đang biên dịch TensorRT Engine..."

        process.wait()
        duration = time.time() - start_time

        if process.returncode == 0 and os.path.exists(output_engine_path):
            engine_size_mb = os.path.getsize(output_engine_path) / (1024 * 1024)
            success_summary = (
                f"✅ **Build TensorRT Engine thành công!**\n"
                f"- **Engine Output**: `{output_engine_path}`\n"
                f"- **Kích thước Engine**: `{engine_size_mb:.2f} MB`\n"
                f"- **Thời gian thực thi**: `{duration:.1f}s`"
            )
            current_log = _append_log(f"\n🎉 SUCCESS: Engine saved to {output_engine_path} ({engine_size_mb:.2f} MB)")
            yield current_log, success_summary
            return {
                "success": True,
                "message": success_summary,
                "engine_path": output_engine_path,
                "duration_seconds": duration,
            }
        else:
            fail_summary = f"❌ **Build Engine thất bại** (Exit code: {process.returncode})!"
            current_log = _append_log(f"\n💥 ERROR: trtexec exited with code {process.returncode}")
            yield current_log, fail_summary
            return {"success": False, "message": fail_summary}

    except Exception as e:
        duration = time.time() - start_time
        err_str = f"❌ Lỗi ngoại lệ khi thực thi trtexec: {type(e).__name__}: {e}"
        logger.exception("Error executing trtexec")
        current_log = _append_log(f"\n💥 EXCEPTION: {err_str}")
        yield current_log, err_str
        return {"success": False, "message": err_str}
