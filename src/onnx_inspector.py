"""
onnx_inspector.py – Đọc và phân tích metadata từ file ONNX

Trích xuất thông tin:
- Input tensors: tên, shape (bao gồm dynamic dims), dtype
- Output tensors: tên, shape, dtype
- Model metadata: opset version, tổng số nodes, IR version
- Yêu cầu calibration data shape dựa trên input specs
"""

import os
from typing import Any

import onnx
import numpy as np


# Mapping ONNX TensorProto.DataType enum → tên đọc được
ONNX_DTYPE_MAP = {
    0: "UNDEFINED",
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}

# Mapping tên dtype → numpy dtype (dùng cho validate calibration data)
DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bfloat16": np.float32,  # numpy không có bfloat16 native, fallback
    "bool": np.bool_,
}


def _extract_tensor_info(tensor) -> dict[str, Any]:
    """Trích xuất thông tin từ một ONNX TensorProto (input hoặc output).

    Args:
        tensor: Một ValueInfoProto từ model.graph.input hoặc model.graph.output.

    Returns:
        Dict chứa name, shape, dtype, is_dynamic.
    """
    name = tensor.name

    # Lấy data type
    elem_type = tensor.type.tensor_type.elem_type
    dtype = ONNX_DTYPE_MAP.get(elem_type, f"unknown({elem_type})")

    # Lấy shape – xử lý cả static và dynamic dimensions
    shape = []
    is_dynamic = False

    tensor_shape = tensor.type.tensor_type.shape
    if tensor_shape is not None:
        for dim in tensor_shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                # Dynamic dimension (ví dụ: "batch_size", "N", "sequence_length")
                shape.append(dim.dim_param)
                is_dynamic = True
            else:
                shape.append("?")
                is_dynamic = True

    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "is_dynamic": is_dynamic,
    }


def inspect_model(onnx_path: str) -> dict[str, Any]:
    """Đọc file ONNX và trả về toàn bộ metadata cần thiết.

    Args:
        onnx_path: Đường dẫn tới file .onnx

    Returns:
        Dict chứa inputs, outputs, opset_version, ir_version, total_nodes,
        producer_name, model_version, và calib_requirements.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file không phải ONNX hợp lệ.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"File ONNX không tồn tại: {onnx_path}")

    if not onnx_path.lower().endswith(".onnx"):
        raise ValueError(f"File không phải định dạng .onnx: {onnx_path}")

    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        raise ValueError(f"Không thể load file ONNX: {e}")

    # Trích xuất inputs (bỏ qua initializers – đó là weights, không phải input thật)
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = []
    for inp in model.graph.input:
        if inp.name not in initializer_names:
            inputs.append(_extract_tensor_info(inp))

    # Trích xuất outputs
    outputs = []
    for out in model.graph.output:
        outputs.append(_extract_tensor_info(out))

    # Opset version
    opset_version = None
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break

    # Đếm tổng số nodes
    total_nodes = len(model.graph.node)

    # Tạo yêu cầu calibration data
    calib_requirements = []
    for inp_info in inputs:
        calib_req = {
            "name": inp_info["name"],
            "expected_shape": inp_info["shape"],
            "expected_dtype": inp_info["dtype"],
            "numpy_dtype": str(DTYPE_TO_NUMPY.get(inp_info["dtype"], np.float32)),
            "note": "",
        }
        if inp_info["is_dynamic"]:
            calib_req["note"] = (
                "Shape có dynamic dimension. Bạn cần cung cấp data với "
                "batch size cụ thể (ví dụ: thay 'batch_size' bằng số lượng "
                "samples thực tế)."
            )
        calib_requirements.append(calib_req)

    return {
        "file_name": os.path.basename(onnx_path),
        "file_size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
        "inputs": inputs,
        "outputs": outputs,
        "opset_version": opset_version,
        "ir_version": model.ir_version,
        "total_nodes": total_nodes,
        "producer_name": model.producer_name or "N/A",
        "model_version": model.model_version,
        "calib_requirements": calib_requirements,
    }


def format_model_info_table(model_info: dict) -> str:
    """Format model info thành chuỗi text đẹp để hiển thị trên Gradio.

    Args:
        model_info: Dict trả về từ inspect_model().

    Returns:
        Chuỗi Markdown formatted.
    """
    lines = []
    lines.append(f"## 📦 Model: `{model_info['file_name']}`")
    lines.append(f"- **Kích thước file**: {model_info['file_size_mb']} MB")
    lines.append(f"- **Opset version**: {model_info['opset_version']}")
    lines.append(f"- **IR version**: {model_info['ir_version']}")
    lines.append(f"- **Producer**: {model_info['producer_name']}")
    lines.append(f"- **Tổng số nodes**: {model_info['total_nodes']}")
    lines.append("")

    # Inputs table
    lines.append("### 📥 Inputs")
    lines.append("| Tên | Shape | Dtype | Dynamic |")
    lines.append("|:---|:---|:---|:---|")
    for inp in model_info["inputs"]:
        shape_str = str(inp["shape"]).replace("'", "")
        dynamic_str = "⚡ Yes" if inp["is_dynamic"] else "No"
        lines.append(f"| `{inp['name']}` | `{shape_str}` | `{inp['dtype']}` | {dynamic_str} |")
    lines.append("")

    # Outputs table
    lines.append("### 📤 Outputs")
    lines.append("| Tên | Shape | Dtype |")
    lines.append("|:---|:---|:---|")
    for out in model_info["outputs"]:
        shape_str = str(out["shape"]).replace("'", "")
        lines.append(f"| `{out['name']}` | `{shape_str}` | `{out['dtype']}` |")
    lines.append("")

    # Calibration requirements
    lines.append("### 🎯 Yêu cầu Calibration Data")
    for req in model_info["calib_requirements"]:
        shape_str = str(req["expected_shape"]).replace("'", "")
        lines.append(f"- **`{req['name']}`**: shape `{shape_str}`, dtype `{req['expected_dtype']}`")
        if req["note"]:
            lines.append(f"  - ⚠️ {req['note']}")

    return "\n".join(lines)
