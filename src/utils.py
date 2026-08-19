"""
utils.py – Tiện ích chung cho QuantTool

Bao gồm:
- Logger setup (console + file)
- Path validation
- Format helpers
"""

import os
import logging
from datetime import datetime


def setup_logger(
    name: str = "quant_tool",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Thiết lập logger ghi ra console và file.

    Args:
        name: Tên logger.
        log_dir: Thư mục chứa file log.
        level: Logging level.

    Returns:
        Logger đã cấu hình.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Tránh duplicate handlers khi gọi lại
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"quant_tool_{timestamp}.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def validate_path(path: str, must_exist: bool = True, extensions: list[str] | None = None) -> tuple[bool, str]:
    """Validate đường dẫn file.

    Args:
        path: Đường dẫn cần validate.
        must_exist: Nếu True, file phải tồn tại.
        extensions: Danh sách extension cho phép (ví dụ: [".onnx"]).

    Returns:
        Tuple (is_valid, message).
    """
    if not path or not path.strip():
        return False, "Đường dẫn rỗng"

    path = path.strip()

    if must_exist and not os.path.exists(path):
        return False, f"File/thư mục không tồn tại: {path}"

    if extensions:
        ext = os.path.splitext(path)[1].lower()
        if ext not in extensions:
            return False, f"Extension '{ext}' không hợp lệ. Chấp nhận: {extensions}"

    return True, "OK"


def format_file_size(size_bytes: int) -> str:
    """Format kích thước file cho dễ đọc.

    Args:
        size_bytes: Kích thước tính bằng bytes.

    Returns:
        Chuỗi formatted (ví dụ: "12.5 MB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def ensure_output_dir(output_path: str) -> str:
    """Đảm bảo thư mục output tồn tại, tạo nếu chưa có.

    Args:
        output_path: Đường dẫn file output.

    Returns:
        Đường dẫn output đã normalize.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_path
