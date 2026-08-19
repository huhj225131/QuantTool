"""
data_validator.py – Validate calibration data so với model input requirements

Hỗ trợ format:
- .npy: Một file numpy array duy nhất
- .npz: Archive chứa nhiều arrays (key = tên input tensor)

Kiểm tra:
- Số lượng inputs khớp
- Shape tương thích (bỏ qua batch dim nếu dynamic)
- Data type tương thích
"""

import os
from typing import Any

import numpy as np

from src.onnx_inspector import DTYPE_TO_NUMPY


def _check_shape_compatible(
    expected_shape: list,
    actual_shape: tuple,
) -> tuple[bool, str]:
    """Kiểm tra shape calibration data có tương thích với model input.

    Logic:
    - Dynamic dims (string hoặc "?") → bỏ qua, chấp nhận bất kỳ giá trị nào
    - Static dims (int) → phải khớp chính xác
    - Nếu expected có N dims và actual có N dims → so sánh từng dim
    - Nếu số dims khác nhau → fail

    Args:
        expected_shape: Shape từ model (có thể chứa string cho dynamic dims).
        actual_shape: Shape thực tế của numpy array.

    Returns:
        Tuple (is_compatible, message).
    """
    if len(expected_shape) != len(actual_shape):
        return False, (
            f"Số chiều không khớp: model yêu cầu {len(expected_shape)}D "
            f"(shape={expected_shape}), data có {len(actual_shape)}D "
            f"(shape={actual_shape})"
        )

    mismatches = []
    for i, (exp, act) in enumerate(zip(expected_shape, actual_shape)):
        if i == 0:
            # Dim 0 luôn biểu diễn Batch Size / Số mẫu Calibration (N) -> Chấp nhận bất kỳ N >= 1
            continue
        if isinstance(exp, str):
            # Dynamic dimension → chấp nhận bất kỳ giá trị
            continue
        if isinstance(exp, int) and exp != act:
            mismatches.append(
                f"  - Dim {i}: model yêu cầu {exp}, data có {act}"
            )

    if mismatches:
        detail = "\n".join(mismatches)
        return False, f"Shape không khớp:\n{detail}"

    return True, "Shape tương thích ✅"


def _check_dtype_compatible(
    expected_dtype: str,
    actual_dtype: np.dtype,
) -> tuple[bool, str]:
    """Kiểm tra dtype có tương thích.

    Cho phép tự động cast trong một số trường hợp an toàn
    (ví dụ: float64 → float32).

    Args:
        expected_dtype: Dtype string từ model (ví dụ: "float32").
        actual_dtype: Numpy dtype thực tế.

    Returns:
        Tuple (is_compatible, message).
    """
    expected_np = DTYPE_TO_NUMPY.get(expected_dtype)
    if expected_np is None:
        return True, f"Dtype '{expected_dtype}' không xác định, bỏ qua kiểm tra"

    # Chấp nhận nếu dtype khớp chính xác
    if actual_dtype == expected_np:
        return True, f"Dtype khớp: {actual_dtype} ✅"

    # Cho phép cast an toàn: float64 → float32, int64 → int32, etc.
    safe_casts = {
        np.float64: [np.float32, np.float16],
        np.float32: [np.float16],
        np.int64: [np.int32, np.int16, np.int8],
        np.int32: [np.int16, np.int8],
    }

    if actual_dtype.type in safe_casts and expected_np in safe_casts.get(actual_dtype.type, []):
        return True, (
            f"Dtype khác ({actual_dtype} → {expected_np.__name__}), "
            f"nhưng có thể cast an toàn ✅"
        )

    # Cho phép upcast float
    if np.issubdtype(actual_dtype, np.floating) and np.issubdtype(expected_np, np.floating):
        return True, (
            f"Dtype khác ({actual_dtype} vs {expected_np.__name__}), "
            f"cả hai đều floating point, có thể cast ✅"
        )

    return False, (
        f"Dtype không tương thích: model yêu cầu {expected_dtype} "
        f"({expected_np.__name__}), data có {actual_dtype}"
    )


def validate_npy_file(
    npy_path: str,
    model_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate một file .npy duy nhất so với model input đầu tiên.

    Args:
        npy_path: Đường dẫn file .npy
        model_inputs: List input info từ inspect_model()["inputs"]

    Returns:
        Dict chứa kết quả validation.
    """
    if not model_inputs:
        return {"valid": False, "message": "Model không có input nào", "details": []}

    try:
        data = np.load(npy_path)
    except Exception as e:
        return {"valid": False, "message": f"Không thể load file .npy: {e}", "details": []}

    if len(model_inputs) > 1:
        return {
            "valid": False,
            "message": (
                f"Model có {len(model_inputs)} inputs, nhưng file .npy chỉ chứa "
                f"1 array. Hãy dùng file .npz với keys tương ứng tên input."
            ),
            "details": [],
        }

    inp = model_inputs[0]
    detail = {"name": inp["name"], "expected_shape": inp["shape"], "actual_shape": list(data.shape)}

    shape_ok, shape_msg = _check_shape_compatible(inp["shape"], data.shape)
    dtype_ok, dtype_msg = _check_dtype_compatible(inp["dtype"], data.dtype)

    detail["shape_check"] = shape_msg
    detail["dtype_check"] = dtype_msg
    detail["actual_dtype"] = str(data.dtype)
    detail["num_samples"] = data.shape[0] if len(data.shape) > 0 else 1

    is_valid = shape_ok and dtype_ok

    return {
        "valid": is_valid,
        "message": "Calibration data hợp lệ ✅" if is_valid else "Calibration data KHÔNG hợp lệ ❌",
        "details": [detail],
    }


def validate_npz_file(
    npz_path: str,
    model_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate file .npz so với toàn bộ model inputs.

    File .npz cần có keys tương ứng với tên các input tensor.

    Args:
        npz_path: Đường dẫn file .npz
        model_inputs: List input info từ inspect_model()["inputs"]

    Returns:
        Dict chứa kết quả validation.
    """
    try:
        data = np.load(npz_path)
    except Exception as e:
        return {"valid": False, "message": f"Không thể load file .npz: {e}", "details": []}

    details = []
    all_valid = True
    available_keys = list(data.keys())

    for inp in model_inputs:
        name = inp["name"]
        detail = {
            "name": name,
            "expected_shape": inp["shape"],
            "expected_dtype": inp["dtype"],
        }

        if name not in data:
            # Thử match theo thứ tự nếu chỉ có 1 input và 1 key
            if len(model_inputs) == 1 and len(available_keys) == 1:
                arr = data[available_keys[0]]
                detail["note"] = (
                    f"Key '{name}' không tìm thấy trong .npz, "
                    f"nhưng dùng key duy nhất '{available_keys[0]}' thay thế"
                )
            else:
                detail["shape_check"] = f"Key '{name}' không tồn tại trong file .npz"
                detail["dtype_check"] = "N/A"
                detail["actual_shape"] = None
                detail["actual_dtype"] = None
                all_valid = False
                details.append(detail)
                continue
        else:
            arr = data[name]

        detail["actual_shape"] = list(arr.shape)
        detail["actual_dtype"] = str(arr.dtype)
        detail["num_samples"] = arr.shape[0] if len(arr.shape) > 0 else 1

        shape_ok, shape_msg = _check_shape_compatible(inp["shape"], arr.shape)
        dtype_ok, dtype_msg = _check_dtype_compatible(inp["dtype"], arr.dtype)

        detail["shape_check"] = shape_msg
        detail["dtype_check"] = dtype_msg

        if not (shape_ok and dtype_ok):
            all_valid = False

        details.append(detail)

    return {
        "valid": all_valid,
        "message": "Calibration data hợp lệ ✅" if all_valid else "Calibration data KHÔNG hợp lệ ❌",
        "details": details,
        "npz_keys": available_keys,
    }


def validate_calibration_data(
    data_path: str,
    model_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Entry point chính – tự động detect format (.npy hoặc .npz) và validate.

    Args:
        data_path: Đường dẫn tới file .npy hoặc .npz
        model_inputs: List input info từ inspect_model()["inputs"]

    Returns:
        Dict chứa valid (bool), message (str), details (list).
    """
    if not os.path.exists(data_path):
        return {
            "valid": False,
            "message": f"File không tồn tại: {data_path}",
            "details": [],
        }

    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".npy":
        return validate_npy_file(data_path, model_inputs)
    elif ext == ".npz":
        return validate_npz_file(data_path, model_inputs)
    else:
        return {
            "valid": False,
            "message": f"Format không hỗ trợ: '{ext}'. Chỉ hỗ trợ .npy và .npz",
            "details": [],
        }


def format_validation_result(result: dict) -> str:
    """Format kết quả validation thành Markdown để hiển thị trên Gradio.

    Args:
        result: Dict trả về từ validate_calibration_data().

    Returns:
        Chuỗi Markdown formatted.
    """
    lines = []
    lines.append(f"## {'✅' if result['valid'] else '❌'} {result['message']}")
    lines.append("")

    if "npz_keys" in result:
        lines.append(f"**Keys trong file .npz**: `{result['npz_keys']}`")
        lines.append("")

    for detail in result["details"]:
        lines.append(f"### Input: `{detail['name']}`")
        lines.append(f"- **Expected shape**: `{detail.get('expected_shape', 'N/A')}`")
        lines.append(f"- **Actual shape**: `{detail.get('actual_shape', 'N/A')}`")
        lines.append(f"- **Expected dtype**: `{detail.get('expected_dtype', 'N/A')}`")
        lines.append(f"- **Actual dtype**: `{detail.get('actual_dtype', 'N/A')}`")

        if "shape_check" in detail:
            icon = "✅" if "✅" in detail["shape_check"] else "❌"
            lines.append(f"- **Shape check**: {icon} {detail['shape_check']}")

        if "dtype_check" in detail and detail["dtype_check"] != "N/A":
            icon = "✅" if "✅" in detail["dtype_check"] else "❌"
            lines.append(f"- **Dtype check**: {icon} {detail['dtype_check']}")

        if "num_samples" in detail:
            lines.append(f"- **Số lượng samples**: {detail['num_samples']}")

        if "note" in detail:
            lines.append(f"- ⚠️ {detail['note']}")

        lines.append("")

    return "\n".join(lines)
