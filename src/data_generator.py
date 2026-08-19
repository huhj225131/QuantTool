"""
data_generator.py – Tiền xử lý dữ liệu thô (ảnh) & tạo tệp Calibration Data (.npy)

Cung cấp 2 phương án:
1. Standard OpenCV Pipeline (Direct Resize, Letterbox/Padding, Center Crop, cv2.dnn.blobFromImage, Normalization)
2. Custom Python Script Execution (Người dùng tự định nghĩa hàm preprocess)
"""

import os
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger("quant_tool.data_generator")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@dataclass
class ImagePreprocessConfig:
    """Cấu hình tham số tiền xử lý ảnh qua OpenCV.

    Attributes:
        target_w: Kích thước Width mong muốn (ví dụ 640).
        target_h: Kích thước Height mong muốn (ví dụ 640).
        resize_mode: Phương pháp resize ("letterbox", "direct", "center_crop", "blob_from_image").
        color_space: Chuyển đổi màu ("bgr2rgb", "keep_bgr", "bgr2gray").
        normalization: Chuẩn hóa ("none_0_255", "scale_0_1", "insightface_minus1_1", "imagenet").
        layout: Thứ tự trục ("NCHW" hoặc "NHWC").
        dtype: Kiểu dữ liệu output ("float32" hoặc "float16").
        pad_value: Giá trị padding cho letterbox (ví dụ 0 hoặc 127.5).
    """

    target_w: int = 640
    target_h: int = 640
    resize_mode: str = "letterbox"
    color_space: str = "bgr2rgb"
    normalization: str = "scale_0_1"
    layout: str = "NCHW"
    dtype: str = "float32"
    pad_value: float = 0.0


def preprocess_image_opencv(
    img_path: str,
    config: ImagePreprocessConfig,
) -> Optional[np.ndarray]:
    """Tiền xử lý 1 tệp ảnh đơn lẻ sử dụng OpenCV theo config.

    Args:
        img_path: Đường dẫn tới file ảnh.
        config: ImagePreprocessConfig chứa toàn bộ tham số.

    Returns:
        Numpy array (C, H, W) hoặc (H, W, C) hoặc None nếu đọc file lỗi.
    """
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Không thể đọc ảnh: {img_path}")
        return None

    target_w, target_h = config.target_w, config.target_h

    # 1. Resize Mode
    if config.resize_mode == "blob_from_image":
        # Sử dụng cv2.dnn.blobFromImage
        swap_rb = (config.color_space == "bgr2rgb")

        # Xác định scalefactor & mean dựa theo normalization
        if config.normalization == "scale_0_1":
            scalefactor = 1.0 / 255.0
            mean = (0, 0, 0)
        elif config.normalization == "insightface_minus1_1":
            scalefactor = 1.0 / 128.0
            mean = (127.5, 127.5, 127.5)
        else:
            scalefactor = 1.0
            mean = (0, 0, 0)

        # blobFromImage tự động resize và đưa về NCHW float32
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=scalefactor,
            size=(target_w, target_h),
            mean=mean,
            swapRB=swap_rb,
            crop=False,
        )
        # Output shape: (1, C, H, W) -> lấy (C, H, W)
        blob = blob[0]

        if config.layout == "NHWC":
            blob = np.transpose(blob, (1, 2, 0))

        if config.dtype == "float16":
            blob = blob.astype(np.float16)

        return blob

    elif config.resize_mode == "letterbox":
        # Letterbox / Top-Left Padding (Det / InsightFace / YOLO style)
        im_ratio = float(img.shape[0]) / float(img.shape[1])
        model_ratio = float(target_h) / float(target_w)

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * im_ratio)

        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.full((target_h, target_w, 3), fill_value=config.pad_value, dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        img_processed = det_img

    elif config.resize_mode == "center_crop":
        # Center Crop (Classification style ResNet / MobileNet)
        h, w = img.shape[:2]
        short_edge = min(h, w)
        crop_x = (w - short_edge) // 2
        crop_y = (h - short_edge) // 2
        cropped = img[crop_y : crop_y + short_edge, crop_x : crop_x + short_edge]
        img_processed = cv2.resize(cropped, (target_w, target_h))

    else:
        # Direct Resize (khớp cứng)
        img_processed = cv2.resize(img, (target_w, target_h))

    # 2. Color Space Conversion
    if config.color_space == "bgr2rgb":
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    elif config.color_space == "bgr2gray":
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        if len(img_processed.shape) == 2:
            img_processed = np.expand_dims(img_processed, axis=-1)

    # 3. Normalization
    img_float = img_processed.astype(np.float32)

    if config.normalization == "scale_0_1":
        img_float /= 255.0
    elif config.normalization == "insightface_minus1_1":
        img_float = (img_float - 127.5) / 128.0
    elif config.normalization == "imagenet":
        img_float /= 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std

    # 4. Layout
    if config.layout == "NCHW" and len(img_float.shape) == 3:
        # (H, W, C) -> (C, H, W)
        img_float = np.transpose(img_float, (2, 0, 1))

    # 5. Dtype
    if config.dtype == "float16":
        img_float = img_float.astype(np.float16)

    return img_float


def scan_image_files(image_dir: str, max_samples: int = 0) -> list[str]:
    """Tìm đệ quy tất cả các file ảnh trong thư mục và các thư mục con.

    Dừng đệ quy sớm ngay khi đạt đủ số lượng max_samples để tối ưu hiệu năng.

    Args:
        image_dir: Thư mục chứa ảnh (quét đệ quy tất cả thư mục con).
        max_samples: Số mẫu ảnh tối đa cần lấy. Nếu > 0, dừng quét ngay khi gom đủ.

    Returns:
        Danh sách đường dẫn ảnh.
    """
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        return []

    files = []
    # Quét đệ quy toàn bộ các thư mục con (recursive walk)
    for root, dirs, filenames in os.walk(image_dir):
        dirs.sort()  # Giữ thứ tự duyệt đệ quy nhất quán
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                files.append(os.path.join(root, fname))
                if 0 < max_samples <= len(files):
                    logger.info(
                        f"⚡ Đã quét đệ quy đủ {max_samples} ảnh từ '{image_dir}', "
                        f"dừng đệ quy sớm để tiết kiệm tài nguyên."
                    )
                    return files

    return files


def generate_calib_dataset_opencv(
    image_dir: str,
    output_npy_path: str,
    config: ImagePreprocessConfig,
    max_samples: int = 100,
    progress_callback=None,
) -> dict[str, Any]:
    """Quét folder ảnh -> tiền xử lý OpenCV -> tạo file .npy calibration dataset.

    Args:
        image_dir: Thư mục chứa ảnh.
        output_npy_path: Đường dẫn lưu file .npy output.
        config: ImagePreprocessConfig.
        max_samples: Số mẫu ảnh tối đa.
        progress_callback: Callback báo tiến độ cho UI (float, msg).

    Returns:
        Dict chứa success (bool), message (str), output_path, shape, file_size_mb.
    """
    def _log(msg, prog=None):
        logger.info(msg)
        if progress_callback and prog is not None:
            progress_callback(prog, msg)

    _log(f"🔍 Đang quét đệ quy ảnh trong: {image_dir} (tối đa {max_samples} ảnh)...", 0.05)
    img_files = scan_image_files(image_dir, max_samples=max_samples)

    if not img_files:
        return {
            "success": False,
            "message": f"❌ Không tìm thấy file ảnh nào (.jpg, .png, .bmp...) trong: {image_dir}",
        }

    if max_samples > 0:
        img_files = img_files[:max_samples]

    total = len(img_files)
    _log(f"📷 Tìm thấy {total} ảnh. Bắt đầu tiền xử lý OpenCV...", 0.1)

    start_time = time.time()
    chunks = []

    for i, img_path in enumerate(img_files):
        arr = preprocess_image_opencv(img_path, config)
        if arr is not None:
            chunks.append(arr)

        if i % 10 == 0 or i == total - 1:
            prog = 0.1 + 0.8 * ((i + 1) / total)
            _log(f"⏳ Đã xử lý {i + 1}/{total} ảnh...", prog)

    if not chunks:
        return {
            "success": False,
            "message": "❌ Tất cả các tệp ảnh đều bị lỗi khi đọc",
        }

    _log("📦 Đang đóng gói batch và lưu file .npy...", 0.95)

    # Concatenate thành batch (N, C, H, W) hoặc (N, H, W, C)
    calib_batch = np.stack(chunks, axis=0)

    # Đảm bảo thư mục lưu tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(output_npy_path)), exist_ok=True)

    np.save(output_npy_path, calib_batch)

    duration = time.time() - start_time
    file_size_mb = os.path.getsize(output_npy_path) / (1024 * 1024)

    _log(f"✅ Đã tạo thành công {output_npy_path}!", 1.0)

    return {
        "success": True,
        "message": (
            f"✅ Tạo Calibration Data `.npy` thành công!\n"
            f"- **Đường dẫn**: `{output_npy_path}`\n"
            f"- **Shape**: `{list(calib_batch.shape)}` (N={calib_batch.shape[0]} samples)\n"
            f"- **Dtype**: `{calib_batch.dtype}`\n"
            f"- **Dung lượng file**: `{file_size_mb:.2f} MB`\n"
            f"- **Thời gian xử lý**: `{duration:.2f}s`"
        ),
        "output_path": output_npy_path,
        "shape": list(calib_batch.shape),
        "dtype": str(calib_batch.dtype),
        "file_size_mb": file_size_mb,
    }


def get_sample_custom_script() -> str:
    """Trả về đoạn code mẫu chuẩn Python cho Custom Script mode."""
    return '''import cv2
import numpy as np

def preprocess_single_image(img_path: str, target_size=(640, 640)) -> np.ndarray:
    """Hàm tiền xử lý 1 ảnh đơn lẻ.

    Args:
        img_path: Đường dẫn tới file ảnh.
        target_size: Tuple (Width, Height) ví dụ (640, 640).

    Returns:
        Numpy array (C, H, W) float32/float16 hoặc None nếu đọc lỗi.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    target_w, target_h = target_size
    im_ratio = float(img.shape[0]) / float(img.shape[1])
    model_ratio = float(target_h) / float(target_w)

    # Top-Left Padding / Letterbox (InsightFace / Det style)
    if im_ratio > model_ratio:
        new_h = target_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = target_w
        new_h = int(new_w * im_ratio)

    resized = cv2.resize(img, (new_w, new_h))
    det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    det_img[:new_h, :new_w, :] = resized

    # BGR -> RGB & Normalize [-1, 1]
    det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    blob = (det_img.astype(np.float32) - 127.5) / 128.0

    # HWC -> CHW
    blob = np.transpose(blob, (2, 0, 1))
    return blob
'''


def load_script_file_content(file_path: str) -> tuple[bool, str]:
    """Đọc nội dung file Python .py.

    Args:
        file_path: Đường dẫn tới tệp .py.

    Returns:
        Tuple (success, content_or_error).
    """
    if not file_path or not file_path.strip():
        return False, "⚠️ Vui lòng nhập đường dẫn file .py"

    file_path = file_path.strip()

    if not os.path.exists(file_path):
        return False, f"❌ File không tồn tại: {file_path}"

    if not file_path.lower().endswith(".py"):
        return False, f"❌ Tệp không phải định dạng Python .py: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        return True, code
    except Exception as e:
        return False, f"❌ Lỗi khi đọc file {file_path}: {e}"


def generate_calib_dataset_custom_script(
    image_dir: str,
    output_npy_path: str,
    script_code: str = "",
    script_file_path: Optional[str] = None,
    target_w: int = 640,
    target_h: int = 640,
    max_samples: int = 100,
    progress_callback=None,
) -> dict[str, Any]:
    """Thực thi mã Python custom của người dùng (từ file .py hoặc code string) để tạo file .npy.

    Args:
        image_dir: Thư mục chứa ảnh.
        output_npy_path: Đường dẫn xuất file .npy.
        script_code: Mã nguồn Python dạng chuỗi text.
        script_file_path: Đường dẫn file .py tùy chọn (nếu có sẽ đọc từ file).
        target_w: Target width.
        target_h: Target height.
        max_samples: Số lượng mẫu tối đa.
        progress_callback: Callback báo tiến độ.

    Returns:
        Dict kết quả.
    """
    def _log(msg, prog=None):
        logger.info(msg)
        if progress_callback and prog is not None:
            progress_callback(prog, msg)

    # Nếu người dùng truyền file .py, ưu tiên đọc nội dung từ file
    if script_file_path and script_file_path.strip():
        ok, file_content = load_script_file_content(script_file_path)
        if not ok:
            return {"success": False, "message": file_content}
        script_code = file_content

    if not script_code or not script_code.strip():
        return {"success": False, "message": "❌ Mã Python custom không được để rỗng (vui lòng chọn file .py hoặc viết code vào ô)"}

    _log("🔍 Đang biên dịch mã Custom Python Script...", 0.05)

    # Dynamic execution môi trường độc lập
    custom_scope = {}
    try:
        exec(script_code, custom_scope)
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ Lỗi cú pháp trong Custom Script:\n{type(e).__name__}: {e}",
        }

    if "preprocess_single_image" not in custom_scope:
        return {
            "success": False,
            "message": (
                "❌ Custom Script thiếu định nghĩa hàm `preprocess_single_image(img_path, target_size)`.\n"
                "Vui lòng nạp lại mẫu code để xem định dạng chuẩn."
            ),
        }

    preprocess_fn = custom_scope["preprocess_single_image"]

    _log(f"🔍 Đang quét đệ quy ảnh trong: {image_dir} (tối đa {max_samples} ảnh)...", 0.1)
    img_files = scan_image_files(image_dir, max_samples=max_samples)

    if not img_files:
        return {
            "success": False,
            "message": f"❌ Không tìm thấy file ảnh nào trong: {image_dir}",
        }

    if max_samples > 0:
        img_files = img_files[:max_samples]

    total = len(img_files)
    _log(f"📷 Tìm thấy {total} ảnh. Đang thực thi Custom Script...", 0.15)

    start_time = time.time()
    chunks = []
    target_size = (target_w, target_h)

    for i, img_path in enumerate(img_files):
        try:
            arr = preprocess_fn(img_path, target_size)
            if arr is not None and isinstance(arr, np.ndarray):
                chunks.append(arr)
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý ảnh {img_path}: {e}")

        if i % 10 == 0 or i == total - 1:
            prog = 0.15 + 0.75 * ((i + 1) / total)
            _log(f"⏳ Đã xử lý {i + 1}/{total} ảnh bằng Custom Script...", prog)

    if not chunks:
        return {
            "success": False,
            "message": "❌ Tất cả các ảnh đều bị lỗi hoặc hàm custom trả về None",
        }

    _log("📦 Đang đóng gói batch và lưu file .npy...", 0.95)

    calib_batch = np.stack(chunks, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_npy_path)), exist_ok=True)
    np.save(output_npy_path, calib_batch)

    duration = time.time() - start_time
    file_size_mb = os.path.getsize(output_npy_path) / (1024 * 1024)

    _log(f"✅ Đã tạo thành công {output_npy_path} qua Custom Script!", 1.0)

    return {
        "success": True,
        "message": (
            f"✅ Tạo Calibration Data bằng Custom Script thành công!\n"
            f"- **Đường dẫn**: `{output_npy_path}`\n"
            f"- **Shape**: `{list(calib_batch.shape)}` (N={calib_batch.shape[0]} samples)\n"
            f"- **Dtype**: `{calib_batch.dtype}`\n"
            f"- **Dung lượng file**: `{file_size_mb:.2f} MB`\n"
            f"- **Thời gian xử lý**: `{duration:.2f}s`"
        ),
        "output_path": output_npy_path,
        "shape": list(calib_batch.shape),
        "dtype": str(calib_batch.dtype),
        "file_size_mb": file_size_mb,
    }
