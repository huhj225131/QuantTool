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


def _dir_contains_target_extension(dir_path: str, extensions: list[str]) -> bool:
    """Kiểm tra nhanh xem thư mục có chứa file thuộc extensions hay không (depth=1).

    Args:
        dir_path: Đường dẫn thư mục.
        extensions: Danh sách extension cần kiểm tra.

    Returns:
        True nếu có ít nhất 1 file phù hợp.
    """
    try:
        ext_set = {e.lower() for e in extensions}
        # Thêm đuôi ảnh nếu extensions rỗng hoặc đang tìm folder ảnh
        if None in extensions or any(e in {".jpg", ".png", ".bmp", ".webp"} for e in extensions):
            ext_set.update({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"})

        for item in os.listdir(dir_path):
            if not item.startswith("."):
                ext = os.path.splitext(item)[1].lower()
                if ext in ext_set:
                    return True
    except (PermissionError, OSError):
        pass
    return False


def list_path_suggestions(
    partial_path: str,
    extensions: list[str] | None = None,
    max_results: int = 30,
) -> list[str]:
    """Gợi ý thông minh đường dẫn file/thư mục dựa trên chuỗi đang gõ.

    Tự động quét và đẩy các thư mục CÓ CHỨA FILE MỤC TIÊU (.onnx, .npy, ảnh...) lên đầu danh sách gợi ý!

    Args:
        partial_path: Chuỗi đường dẫn đang gõ (ví dụ: "/home/user/mo").
        extensions: Lọc theo extension (ví dụ: [".onnx", ".npy"]).
                   Nếu None, hiển thị tất cả. Thư mục chứa file phù hợp sẽ được ưu tiên.
        max_results: Số kết quả tối đa trả về.

    Returns:
        Danh sách đường dẫn gợi ý (thư mục có suffix "/").
    """
    if not partial_path or not partial_path.strip():
        # Mặc định hiển thị thư mục cha (các file/folder ngang hàng với project)
        parent_dir = os.path.dirname(os.getcwd())
        prefix = ""
    else:
        partial_path = partial_path.strip()
        partial_path = os.path.expanduser(partial_path)

        if os.path.isdir(partial_path):
            parent_dir = partial_path
            prefix = ""
        else:
            parent_dir = os.path.dirname(partial_path)
            prefix = os.path.basename(partial_path).lower()

    if not os.path.isdir(parent_dir):
        return []

    try:
        entries = sorted(os.listdir(parent_dir))
    except PermissionError:
        return [f"⚠️ Không có quyền truy cập: {parent_dir}"]

    priority_dirs = []
    normal_dirs = []
    matching_files = []

    for entry in entries:
        # Bỏ qua file/thư mục ẩn (bắt đầu bằng .)
        if entry.startswith("."):
            continue

        # Filter theo prefix đang gõ
        if prefix and not entry.lower().startswith(prefix):
            continue

        full_path = os.path.join(parent_dir, entry)

        if os.path.isdir(full_path):
            dir_path = full_path + "/"
            # Kiểm tra xem thư mục có chứa file mục tiêu không -> Đẩy lên ưu tiên!
            if extensions and _dir_contains_target_extension(full_path, extensions):
                priority_dirs.append(dir_path)
            else:
                normal_dirs.append(dir_path)
        elif extensions:
            # File: chỉ hiển thị nếu extension khớp
            _, ext = os.path.splitext(entry)
            if ext.lower() in extensions:
                matching_files.append(full_path)
        else:
            # Không filter extension → hiển thị tất cả file
            matching_files.append(full_path)

    # Sắp xếp thứ tự ưu tiên: Thư mục chứa target files -> Các thư mục còn lại -> Các file khớp
    ordered_suggestions = priority_dirs + normal_dirs + matching_files
    return ordered_suggestions[:max_results]

