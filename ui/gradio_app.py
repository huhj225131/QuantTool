"""
gradio_app.py – Giao diện Gradio cho QuantTool

3 Tabs chính:
1. Model Inspector: Nhập/chọn đường dẫn ONNX → hiển thị metadata + shape requirements
2. Calibration Data:
   - Accordion 1: Tạo dữ liệu Calibration (.npy) từ ảnh thô (OpenCV Pipeline / Custom Python Script)
   - Accordion 2: Validate Calibration Data (.npy / .npz) so với model
3. Quantization: Nhập/chọn ONNX model path, Calib data path, output path + cấu hình tham số → quantize
"""

import os
import gradio as gr

from src.onnx_inspector import inspect_model, format_model_info_table
from src.data_validator import validate_calibration_data, format_validation_result
from src.quantizer import QuantConfig, quantize_model, check_modelopt_available
from src.data_generator import (
    ImagePreprocessConfig,
    generate_calib_dataset_opencv,
    generate_calib_dataset_custom_script,
    get_sample_custom_script,
)
from src.utils import setup_logger, list_path_suggestions

logger = setup_logger()

# State lưu trữ thông tin model đang inspect
_current_model_cache = {}


# ═══════════════════════════════════════════════════════════
# Path autocomplete callbacks (Dùng trực tiếp trên Dropdown)
# ═══════════════════════════════════════════════════════════

def suggest_onnx_paths(val):
    """Gợi ý đường dẫn file .onnx và thư mục trực tiếp trên dropdown."""
    suggestions = list_path_suggestions(val, extensions=[".onnx"])
    return gr.update(choices=suggestions)


def suggest_calib_paths(val):
    """Gợi ý đường dẫn file .npy/.npz và thư mục trực tiếp trên dropdown."""
    suggestions = list_path_suggestions(val, extensions=[".npy", ".npz"])
    return gr.update(choices=suggestions)


def suggest_all_dirs(val):
    """Gợi ý đường dẫn thư mục."""
    suggestions = list_path_suggestions(val, extensions=None)
    return gr.update(choices=suggestions)


def _get_or_inspect_model(onnx_path: str):
    """Helper lấy model_info từ cache hoặc inspect mới."""
    onnx_path = onnx_path.strip()
    if onnx_path in _current_model_cache:
        return _current_model_cache[onnx_path]

    model_info = inspect_model(onnx_path)
    _current_model_cache[onnx_path] = model_info
    return model_info


# ═══════════════════════════════════════════════════════════
# Main action callbacks
# ═══════════════════════════════════════════════════════════

def on_inspect_model(onnx_path):
    """Callback khi user inspect model ở Tab 1."""
    if not onnx_path or not onnx_path.strip():
        return (
            "⚠️ Vui lòng nhập hoặc chọn đường dẫn file ONNX",
            "",
            gr.update(),  # calib_model_path (Tab 2)
            gr.update(),  # quant_model_path (Tab 3)
            gr.update(),  # target_w (Tab 2)
            gr.update(),  # target_h (Tab 2)
            gr.update(),  # target_w_custom (Tab 2)
            gr.update(),  # target_h_custom (Tab 2)
        )

    onnx_path = onnx_path.strip()

    if not os.path.exists(onnx_path):
        return (
            f"❌ File không tồn tại: `{onnx_path}`",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        model_info = _get_or_inspect_model(onnx_path)

        # Format hiển thị
        info_text = format_model_info_table(model_info)

        # Format calibration requirements
        calib_text = "### 🎯 Yêu cầu Calibration Data\n\n"
        for req in model_info["calib_requirements"]:
            shape_str = str(req["expected_shape"]).replace("'", "")
            calib_text += f"- **`{req['name']}`**: shape `{shape_str}`, dtype `{req['expected_dtype']}`\n"
            if req["note"]:
                calib_text += f"  - ⚠️ {req['note']}\n"

        logger.info(f"Model inspected: {model_info['file_name']} ({model_info['total_nodes']} nodes)")

        # Thử lấy W, H từ input 0 nếu có
        w_update = gr.update()
        h_update = gr.update()
        if model_info["inputs"]:
            shape = model_info["inputs"][0]["shape"]
            # ví dụ [1, 3, 640, 640] -> H=shape[2], W=shape[3] hoặc [1, 640, 640, 3] -> H=shape[1], W=shape[2]
            int_dims = [d for d in shape if isinstance(d, int)]
            if len(int_dims) >= 2:
                # Thường H, W là 2 dims cuối hoặc 2 dims sau channel
                h_val = int_dims[-2] if int_dims[-2] > 10 else int_dims[-1]
                w_val = int_dims[-1] if int_dims[-1] > 10 else int_dims[-2]
                w_update = gr.update(value=w_val)
                h_update = gr.update(value=h_val)

        # Tự động đồng bộ đường dẫn model sang Tab 2 và Tab 3
        return (
            info_text,
            calib_text,
            gr.update(value=onnx_path),
            gr.update(value=onnx_path),
            w_update,
            h_update,
            w_update,
            h_update,
        )

    except Exception as e:
        logger.exception("Error inspecting model")
        return (
            f"❌ Lỗi khi đọc model: {e}",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def on_validate_data(onnx_path, data_path):
    """Callback khi user validate calibration data ở Tab 2."""
    if not onnx_path or not onnx_path.strip():
        return "⚠️ Vui lòng nhập hoặc chọn đường dẫn file ONNX model", gr.update()

    if not data_path or not data_path.strip():
        return "⚠️ Vui lòng nhập hoặc chọn đường dẫn calibration data", gr.update()

    onnx_path = onnx_path.strip()
    data_path = data_path.strip()

    if not os.path.exists(onnx_path):
        return f"❌ File ONNX không tồn tại: `{onnx_path}`", gr.update()

    if not os.path.exists(data_path):
        return f"❌ File calibration data không tồn tại: `{data_path}`", gr.update()

    try:
        model_info = _get_or_inspect_model(onnx_path)
        model_inputs = model_info["inputs"]

        result = validate_calibration_data(data_path, model_inputs)
        formatted = format_validation_result(result)
        logger.info(f"Validation result for {data_path}: {result['message']}")

        # Đồng bộ calib data path sang Tab 3 nếu hợp lệ
        sync_calib_update = gr.update(value=data_path) if result["valid"] else gr.update()

        return formatted, sync_calib_update

    except Exception as e:
        logger.exception("Error validating data")
        return f"❌ Lỗi khi validate: {e}", gr.update()


def on_generate_opencv(
    image_dir,
    output_npy_path,
    target_w,
    target_h,
    resize_mode,
    color_space,
    normalization,
    layout,
    dtype,
    max_samples,
    progress=gr.Progress(),
):
    """Callback chạy generator OpenCV."""
    if not image_dir or not image_dir.strip():
        return "⚠️ Vui lòng nhập hoặc chọn thư mục chứa ảnh thô", gr.update(), gr.update()

    if not output_npy_path or not output_npy_path.strip():
        output_npy_path = "calib_data/calib_dataset.npy"

    resize_mode_map = {
        "Letterbox / Top-Left Padding (InsightFace/Det Style)": "letterbox",
        "Direct Resize (Ép cứng)": "direct",
        "Center Crop (Classification Style)": "center_crop",
        "OpenCV DNN Blob (cv2.dnn.blobFromImage)": "blob_from_image",
    }
    color_space_map = {
        "BGR -> RGB (PyTorch/ONNX)": "bgr2rgb",
        "BGR (Keep OpenCV Mặc định)": "keep_bgr",
        "BGR -> GRAY (Ảnh xám)": "bgr2gray",
    }
    normalization_map = {
        "[0.0, 1.0] (Scale 1/255.0)": "scale_0_1",
        "[-1.0, 1.0] (InsightFace / (img - 127.5) / 128.0)": "insightface_minus1_1",
        "ImageNet Z-score (mean/std)": "imagenet",
        "[0, 255] (Keep uint8/float32 raw)": "none_0_255",
    }

    config = ImagePreprocessConfig(
        target_w=int(target_w),
        target_h=int(target_h),
        resize_mode=resize_mode_map.get(resize_mode, "letterbox"),
        color_space=color_space_map.get(color_space, "bgr2rgb"),
        normalization=normalization_map.get(normalization, "scale_0_1"),
        layout=layout,
        dtype=dtype,
    )

    def progress_callback(prog, msg):
        progress(prog, desc=msg)

    result = generate_calib_dataset_opencv(
        image_dir=image_dir.strip(),
        output_npy_path=output_npy_path.strip(),
        config=config,
        max_samples=int(max_samples),
        progress_callback=progress_callback,
    )

    if result["success"]:
        sync_calib = gr.update(value=result["output_path"])
        return result["message"], sync_calib, sync_calib
    else:
        return result["message"], gr.update(), gr.update()


def suggest_py_paths(val):
    """Gợi ý đường dẫn file .py và thư mục trực tiếp trên dropdown."""
    suggestions = list_path_suggestions(val, extensions=[".py"])
    return gr.update(choices=suggestions)


def on_load_file_to_editor(file_path):
    """Nạp nội dung từ file .py trên đĩa vào ô Code Editor."""
    from src.data_generator import load_script_file_content
    ok, content = load_script_file_content(file_path)
    if ok:
        return content, f"✅ Đã nạp thành công mã nguồn từ: `{file_path}`"
    else:
        return gr.update(), content


def on_generate_custom_script(
    image_dir,
    output_npy_path,
    script_code,
    script_file_path,
    target_w,
    target_h,
    max_samples,
    progress=gr.Progress(),
):
    """Callback chạy Custom Python Script generator."""
    if not image_dir or not image_dir.strip():
        return "⚠️ Vui lòng nhập hoặc chọn thư mục chứa ảnh thô", gr.update(), gr.update()

    if not output_npy_path or not output_npy_path.strip():
        output_npy_path = "calib_data/calib_custom.npy"

    def progress_callback(prog, msg):
        progress(prog, desc=msg)

    result = generate_calib_dataset_custom_script(
        image_dir=image_dir.strip(),
        output_npy_path=output_npy_path.strip(),
        script_code=script_code,
        script_file_path=script_file_path.strip() if script_file_path else None,
        target_w=int(target_w),
        target_h=int(target_h),
        max_samples=int(max_samples),
        progress_callback=progress_callback,
    )

    if result["success"]:
        sync_calib = gr.update(value=result["output_path"])
        return result["message"], sync_calib, sync_calib
    else:
        return result["message"], gr.update(), gr.update()


def on_load_sample_script():
    """Callback nạp lại template code mẫu Custom Script."""
    return get_sample_custom_script(), "✅ Đã nạp lại mẫu code Python chuẩn"


def on_start_quantize(
    onnx_path,
    calib_data_path,
    quantize_mode,
    calibration_method,
    output_path,
    progress=gr.Progress(),
):
    """Callback khi user bấm Start Quantization ở Tab 3."""
    if not onnx_path or not onnx_path.strip():
        return "⚠️ Vui lòng nhập hoặc chọn đường dẫn file ONNX model"

    if not calib_data_path or not calib_data_path.strip():
        return "⚠️ Vui lòng nhập hoặc chọn đường dẫn calibration data"

    onnx_path = onnx_path.strip()
    calib_data_path = calib_data_path.strip()

    if not os.path.exists(onnx_path):
        return f"⚠️ File ONNX không tồn tại: `{onnx_path}`"

    if not os.path.exists(calib_data_path):
        return f"⚠️ File calibration data không tồn tại: `{calib_data_path}`"

    if not output_path.strip():
        output_path = "outputs/model_quantized.onnx"

    config = QuantConfig(
        onnx_path=onnx_path,
        output_path=output_path.strip(),
        quantize_mode=quantize_mode,
        calibration_method=calibration_method,
        calibration_data_path=calib_data_path,
    )

    def progress_callback(prog, msg):
        progress(prog, desc=msg)

    result = quantize_model(config, progress_callback=progress_callback)

    return result["message"]


def on_check_modelopt():
    """Kiểm tra nvidia-modelopt có sẵn không."""
    available, msg = check_modelopt_available()
    return msg


def on_quant_mode_change(quant_mode):
    """Cập nhật danh sách calibration_method phù hợp theo quantize_mode.

    - FP8 & INT8: max, entropy
    - INT4: awq_clip, rtn_dq
    """
    if quant_mode in ["fp8", "int8"]:
        return gr.update(
            choices=["max", "entropy"],
            value="max",
            info="max: khuyến nghị cho TensorRT. entropy: KL divergence calibration.",
        )
    elif quant_mode == "int4":
        return gr.update(
            choices=["awq_clip", "rtn_dq"],
            value="awq_clip",
            info="awq_clip: Weight-Only AWQ calibration. rtn_dq: Round-To-Nearest Dequantize.",
        )
    return gr.update()


# ═══════════════════════════════════════════════════════════
# UI Builder
# ═══════════════════════════════════════════════════════════

def create_ui() -> gr.Blocks:
    """Tạo Gradio Blocks UI chính."""

    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    """

    with gr.Blocks(
        title="QuantTool – ONNX Model Optimizer",
    ) as app:
        # Header
        gr.HTML("""
            <div class="main-title">⚡ QuantTool</div>
            <div class="subtitle">ONNX Model Optimization powered by NVIDIA ModelOpt</div>
        """)

        with gr.Tabs() as tabs:
            # ═══════════════════════════════════════════════════════════
            # TAB 1: Model Inspector
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📦 Model Inspector", id="tab_inspector"):
                gr.Markdown("### Chọn hoặc nhập đường dẫn file ONNX để phân tích cấu trúc model")

                with gr.Row():
                    with gr.Column(scale=3):
                        onnx_path_input = gr.Dropdown(
                            choices=list_path_suggestions("", extensions=[".onnx"]),
                            value="",
                            label="📁 Đường dẫn ONNX Model",
                            info="Gõ đường dẫn để hiện gợi ý trực tiếp ngay tại đây (hỗ trợ ~ cho Home directory)",
                            allow_custom_value=True,
                            filterable=True,
                        )
                    with gr.Column(scale=1):
                        inspect_btn = gr.Button(
                            "🔍 Inspect Model",
                            variant="primary",
                            size="lg",
                        )

                model_info_display = gr.Markdown(
                    value="*Chưa có model nào được inspect...*",
                    label="Model Information",
                )

                calib_req_display = gr.Markdown(
                    value="",
                    label="Calibration Requirements",
                    visible=True,
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 2: Calibration Data (Generator & Validator)
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📊 Calibration Data", id="tab_calib"):
                gr.Markdown("### Tiền xử lý dữ liệu thô (tạo file .npy) & Validate dữ liệu Calibration")

                # --- ACCORDION 1: Calibration Data Generator ---
                with gr.Accordion("🛠️ 1. Tạo file Calibration Data (.npy) từ ảnh thô", open=True):
                    with gr.Tabs() as gen_tabs:
                        # --- TAB A: OpenCV Pipeline ---
                        with gr.TabItem("⚙️ OpenCV Pipeline (Chuẩn)", id="gen_tab_opencv"):
                            with gr.Row():
                                img_dir_opencv = gr.Dropdown(
                                    choices=list_path_suggestions("", extensions=None),
                                    value="",
                                    label="📁 Thư mục chứa ảnh thô",
                                    info="Thư mục chứa các tệp ảnh (.jpg, .png, .bmp...)",
                                    allow_custom_value=True,
                                    filterable=True,
                                )
                                output_npy_opencv = gr.Textbox(
                                    label="💾 Đường dẫn lưu file .npy",
                                    value="calib_data/calib_dataset.npy",
                                    info="Tệp .npy tạo ra sẽ được lưu tại đây",
                                )

                            with gr.Row():
                                target_w = gr.Number(value=640, label="Target Width (pixels)", precision=0)
                                target_h = gr.Number(value=640, label="Target Height (pixels)", precision=0)
                                max_samples_opencv = gr.Number(value=100, label="Số ảnh tối đa (Max Samples)", precision=0)

                            with gr.Row():
                                resize_mode_dropdown = gr.Dropdown(
                                    choices=[
                                        "Letterbox / Top-Left Padding (InsightFace/Det Style)",
                                        "Direct Resize (Ép cứng)",
                                        "Center Crop (Classification Style)",
                                        "OpenCV DNN Blob (cv2.dnn.blobFromImage)",
                                    ],
                                    value="Letterbox / Top-Left Padding (InsightFace/Det Style)",
                                    label="Resize Mode & Padding",
                                )
                                color_space_dropdown = gr.Dropdown(
                                    choices=[
                                        "BGR -> RGB (PyTorch/ONNX)",
                                        "BGR (Keep OpenCV Mặc định)",
                                        "BGR -> GRAY (Ảnh xám)",
                                    ],
                                    value="BGR -> RGB (PyTorch/ONNX)",
                                    label="Color Space Conversion",
                                )

                            with gr.Row():
                                norm_dropdown = gr.Dropdown(
                                    choices=[
                                        "[0.0, 1.0] (Scale 1/255.0)",
                                        "[-1.0, 1.0] (InsightFace / (img - 127.5) / 128.0)",
                                        "ImageNet Z-score (mean/std)",
                                        "[0, 255] (Keep uint8/float32 raw)",
                                    ],
                                    value="[0.0, 1.0] (Scale 1/255.0)",
                                    label="Normalization Mode",
                                )
                                layout_dropdown = gr.Dropdown(
                                    choices=["NCHW", "NHWC"],
                                    value="NCHW",
                                    label="Layout (NCHW / NHWC)",
                                )
                                dtype_dropdown = gr.Dropdown(
                                    choices=["float32", "float16"],
                                    value="float32",
                                    label="Data Type (dtype)",
                                )

                            gen_opencv_btn = gr.Button(
                                "⚡ Generate Calibration File (.npy) via OpenCV",
                                variant="primary",
                                size="lg",
                            )

                        # --- TAB B: Custom Python Script ---
                        with gr.TabItem("🐍 Custom Python Script", id="gen_tab_custom"):
                            gr.Markdown(
                                "Tự định nghĩa hàm `preprocess_single_image(img_path, target_size)` bằng Python. "
                                "Hệ thống sẽ thực thi hàm này trên toàn bộ ảnh trong thư mục để đóng gói thành tệp `.npy`."
                            )
                            with gr.Row():
                                img_dir_custom = gr.Dropdown(
                                    choices=list_path_suggestions("", extensions=None),
                                    value="",
                                    label="📁 Thư mục chứa ảnh thô",
                                    info="Thư mục chứa các tệp ảnh",
                                    allow_custom_value=True,
                                    filterable=True,
                                )
                                output_npy_custom = gr.Textbox(
                                    label="💾 Đường dẫn lưu file .npy",
                                    value="calib_data/calib_custom.npy",
                                )

                            with gr.Row():
                                target_w_custom = gr.Number(value=640, label="Target Width", precision=0)
                                target_h_custom = gr.Number(value=640, label="Target Height", precision=0)
                                max_samples_custom = gr.Number(value=100, label="Số ảnh tối đa", precision=0)

                            with gr.Row():
                                with gr.Column(scale=3):
                                    script_file_path = gr.Dropdown(
                                        choices=list_path_suggestions("", extensions=[".py"]),
                                        value="",
                                        label="📄 Đường dẫn file Python Script (.py) trên đĩa",
                                        info="Chọn hoặc gõ đường dẫn file .py có sẵn (nếu có sẽ ưu tiên thực thi file này)",
                                        allow_custom_value=True,
                                        filterable=True,
                                    )
                                with gr.Column(scale=1):
                                    load_file_btn = gr.Button("📂 Nạp code từ file .py", size="sm")

                            with gr.Row():
                                script_editor = gr.Code(
                                    value=get_sample_custom_script(),
                                    language="python",
                                    label="📝 Trình soạn thảo Python Script (Hoặc chỉnh sửa trực tiếp tại đây)",
                                    lines=18,
                                )

                            with gr.Row():
                                reset_script_btn = gr.Button("🔄 Nạp lại Code Mẫu", size="sm")
                                gen_custom_btn = gr.Button(
                                    "⚡ Run Custom Script & Save .npy",
                                    variant="primary",
                                    size="lg",
                                )

                    gen_result_display = gr.Markdown(
                        value="*Chưa tạo dữ liệu calibration...*",
                        label="Kết quả Tạo Calibration Data",
                    )

                gr.Markdown("---")

                # --- ACCORDION 2: Calibration Data Validator ---
                with gr.Accordion("🔍 2. Kiểm tra tính hợp lệ Calibration Data (Validator)", open=True):
                    with gr.Row():
                        with gr.Column(scale=3):
                            calib_model_path = gr.Dropdown(
                                choices=list_path_suggestions("", extensions=[".onnx"]),
                                value="",
                                label="📦 Đường dẫn ONNX Model",
                                info="Model cần validate calibration data (tự động điền từ Tab 1)",
                                allow_custom_value=True,
                                filterable=True,
                            )

                            calib_path_input = gr.Dropdown(
                                choices=list_path_suggestions("", extensions=[".npy", ".npz"]),
                                value="",
                                label="📊 Đường dẫn Calibration Data (.npy / .npz)",
                                info="Đường dẫn file .npy/.npz (tự động điền từ Bước 1 khi tạo xong)",
                                allow_custom_value=True,
                                filterable=True,
                            )
                        with gr.Column(scale=1):
                            validate_btn = gr.Button(
                                "🔍 Validate Data",
                                variant="primary",
                                size="lg",
                            )

                    validation_result = gr.Markdown(
                        value="*Chưa validate...*",
                        label="Kết quả Validation",
                    )

            # ═══════════════════════════════════════════════════════════
            # TAB 3: Quantization
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("🚀 Quantization", id="tab_quant"):
                gr.Markdown("### Cấu hình và chạy quantization")

                # Check ModelOpt availability
                with gr.Row():
                    check_btn = gr.Button("🔧 Kiểm tra ModelOpt", size="sm")
                    modelopt_status = gr.Textbox(
                        label="Trạng thái ModelOpt",
                        interactive=False,
                        value="Chưa kiểm tra...",
                    )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        quant_model_path = gr.Dropdown(
                            choices=list_path_suggestions("", extensions=[".onnx"]),
                            value="",
                            label="📦 Đường dẫn ONNX Model gốc",
                            info="Model cần quantize (tự động đồng bộ từ Tab 1 & Tab 2)",
                            allow_custom_value=True,
                            filterable=True,
                        )

                        quant_calib_path = gr.Dropdown(
                            choices=list_path_suggestions("", extensions=[".npy", ".npz"]),
                            value="",
                            label="📊 Đường dẫn Calibration Data (.npy / .npz)",
                            info="Dữ liệu calibration (tự động đồng bộ từ Tab 2 khi validate thành công)",
                            allow_custom_value=True,
                            filterable=True,
                        )

                    with gr.Column():
                        quant_mode_dropdown = gr.Dropdown(
                            choices=["int8", "fp8", "int4"],
                            value="int8",
                            label="Quantize Mode",
                            info="int8: Phổ biến & cân bằng. fp8: Cho GPU Hopper/Ada (CC >= 8.9). int4: Weight-Only AWQ/RTN.",
                        )

                        calib_method_dropdown = gr.Dropdown(
                            choices=["max", "entropy"],
                            value="max",
                            label="Calibration Method",
                            info="max: khuyến nghị cho TensorRT. entropy: KL divergence calibration.",
                        )

                        quant_output_path = gr.Textbox(
                            label="💾 Đường dẫn Output Model",
                            value="outputs/model_quantized.onnx",
                            info="Model đã quantize sẽ được lưu tại đây",
                        )

                with gr.Accordion("📖 Hướng dẫn & Tối ưu hóa NVIDIA ModelOpt cho ONNX", open=False):
                    gr.Markdown("""
                    #### 🎯 Hướng dẫn lựa chọn chế độ Tối ưu (Quantization Schemes)

                    - **`INT8`**:
                      - **Calibration Method**: `max` (khuyến nghị cho TensorRT) hoặc `entropy` (KL divergence).
                      - **Ứng dụng**: Phù hợp với đa số mô hình Vision (ResNet, YOLO, ViT...). Cân bằng tốt giữa tốc độ suy luận và độ chính xác.

                    - **`FP8`**:
                      - **Calibration Method**: `max` hoặc `entropy`.
                      - **Yêu cầu Phần cứng**: Cần GPU thế hệ **NVIDIA Hopper** hoặc **Ada Lovelace** có Compute Capability $\ge 8.9$ (ví dụ: RTX 40xx series, H100, L40, L4).

                    - **`INT4 (AWQ / RTN)`**:
                      - **Calibration Method**: `awq_clip` (Activation-aware Weight Quantization) hoặc `rtn_dq` (Round-To-Nearest Dequantize).
                      - **Ứng dụng**: **Weight-Only Quantization**. Rất hiệu quả cho **low-batch inference** nơi thời gian suy luận bị giới hạn bởi tốc độ đọc weights từ VRAM/RAM (memory bandwidth bound). Giúp giảm độ trễ thấp hơn FP8/INT8 ở batch nhỏ và bảo toàn độ chính xác tốt hơn INT8 thuần.

                    ---
                    #### 💻 Lệnh CLI tương đương:
                    ```bash
                    python -m modelopt.onnx.quantization \\
                        --onnx_path=model.onnx \\
                        --quantize_mode=<fp8|int8|int4> \\
                        --calibration_data=calib.npy \\
                        --calibration_method=<max|entropy|awq_clip|rtn_dq> \\
                        --output_path=model.quant.onnx
                    ```
                    """)

                with gr.Row():
                    quantize_btn = gr.Button(
                        "⚡ Start Quantization",
                        variant="primary",
                        size="lg",
                    )

                quant_result = gr.Markdown(
                    value="*Chưa chạy quantization...*",
                    label="Kết quả Quantization",
                )

        # ═══════════════════════════════════════════════════════════
        # Event handlers (Autocomplete trực tiếp trên Dropdown)
        # ═══════════════════════════════════════════════════════════

        # Tab 1: Autocomplete ONNX path
        onnx_path_input.input(
            fn=suggest_onnx_paths,
            inputs=[onnx_path_input],
            outputs=[onnx_path_input],
        )

        # Tab 2 Generator: Autocomplete folder paths
        img_dir_opencv.input(
            fn=suggest_all_dirs,
            inputs=[img_dir_opencv],
            outputs=[img_dir_opencv],
        )
        img_dir_custom.input(
            fn=suggest_all_dirs,
            inputs=[img_dir_custom],
            outputs=[img_dir_custom],
        )

        script_file_path.input(
            fn=suggest_py_paths,
            inputs=[script_file_path],
            outputs=[script_file_path],
        )

        load_file_btn.click(
            fn=on_load_file_to_editor,
            inputs=[script_file_path],
            outputs=[script_editor, gen_result_display],
        )

        # Tab 2 Validator: Autocomplete Model & Calib paths
        calib_model_path.input(
            fn=suggest_onnx_paths,
            inputs=[calib_model_path],
            outputs=[calib_model_path],
        )
        calib_path_input.input(
            fn=suggest_calib_paths,
            inputs=[calib_path_input],
            outputs=[calib_path_input],
        )

        # Tab 3: Autocomplete Model & Calib paths
        quant_model_path.input(
            fn=suggest_onnx_paths,
            inputs=[quant_model_path],
            outputs=[quant_model_path],
        )
        quant_calib_path.input(
            fn=suggest_calib_paths,
            inputs=[quant_calib_path],
            outputs=[quant_calib_path],
        )

        # Tab 2: Generator OpenCV action -> Sync generated .npy to Tab 2 Validator & Tab 3 Quantization
        gen_opencv_btn.click(
            fn=on_generate_opencv,
            inputs=[
                img_dir_opencv,
                output_npy_opencv,
                target_w,
                target_h,
                resize_mode_dropdown,
                color_space_dropdown,
                norm_dropdown,
                layout_dropdown,
                dtype_dropdown,
                max_samples_opencv,
            ],
            outputs=[gen_result_display, calib_path_input, quant_calib_path],
        )

        # Tab 2: Generator Custom Script action
        gen_custom_btn.click(
            fn=on_generate_custom_script,
            inputs=[
                img_dir_custom,
                output_npy_custom,
                script_editor,
                script_file_path,
                target_w_custom,
                target_h_custom,
                max_samples_custom,
            ],
            outputs=[gen_result_display, calib_path_input, quant_calib_path],
        )

        # Tab 2: Reset custom script sample code
        reset_script_btn.click(
            fn=on_load_sample_script,
            inputs=[],
            outputs=[script_editor, gen_result_display],
        )

        # Dynamic Quantize Mode -> Calibration Method choices
        quant_mode_dropdown.change(
            fn=on_quant_mode_change,
            inputs=[quant_mode_dropdown],
            outputs=[calib_method_dropdown],
        )

        # Inspect button (Tab 1) -> Trả về info + Sync model path & dimensions tới Tab 2 & 3
        inspect_btn.click(
            fn=on_inspect_model,
            inputs=[onnx_path_input],
            outputs=[
                model_info_display,
                calib_req_display,
                calib_model_path,
                quant_model_path,
                target_w,
                target_h,
                target_w_custom,
                target_h_custom,
            ],
        )

        # Validate button (Tab 2) -> Validate + Sync calib path tới Tab 3
        validate_btn.click(
            fn=on_validate_data,
            inputs=[calib_model_path, calib_path_input],
            outputs=[validation_result, quant_calib_path],
        )

        # Check modelopt (Tab 3)
        check_btn.click(
            fn=on_check_modelopt,
            inputs=[],
            outputs=[modelopt_status],
        )

        # Start quantization (Tab 3)
        quantize_btn.click(
            fn=on_start_quantize,
            inputs=[
                quant_model_path,
                quant_calib_path,
                quant_mode_dropdown,
                calib_method_dropdown,
                quant_output_path,
            ],
            outputs=[quant_result],
        )

    # Lưu theme/css để truyền vào launch() (Gradio 6.0+ API)
    app._custom_css = custom_css
    app._custom_theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="slate",
    )
    return app
