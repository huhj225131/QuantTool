"""
gradio_app.py – Giao diện Gradio cho QuantTool

3 Tabs chính:
1. Model Inspector: Upload ONNX → hiển thị metadata + shape requirements
2. Calibration Data: Validate calibration data so với model
3. Quantization: Cấu hình tham số + chạy quantize
"""

import os
import gradio as gr

from src.onnx_inspector import inspect_model, format_model_info_table
from src.data_validator import validate_calibration_data, format_validation_result
from src.quantizer import QuantConfig, quantize_model, check_modelopt_available
from src.utils import setup_logger

logger = setup_logger()

# State lưu trữ thông tin model đang inspect
_current_model_info = {"data": None}


def on_upload_model(file):
    """Callback khi user upload file ONNX."""
    if file is None:
        return "⚠️ Vui lòng upload file ONNX", ""

    try:
        model_info = inspect_model(file.name)
        _current_model_info["data"] = model_info

        # Format hiển thị
        info_text = format_model_info_table(model_info)

        # Format calibration requirements riêng cho tab 2
        calib_text = "### 🎯 Yêu cầu Calibration Data\n\n"
        for req in model_info["calib_requirements"]:
            shape_str = str(req["expected_shape"]).replace("'", "")
            calib_text += f"- **`{req['name']}`**: shape `{shape_str}`, dtype `{req['expected_dtype']}`\n"
            if req["note"]:
                calib_text += f"  - ⚠️ {req['note']}\n"

        logger.info(f"Model loaded: {model_info['file_name']} ({model_info['total_nodes']} nodes)")
        return info_text, calib_text

    except Exception as e:
        logger.exception("Error inspecting model")
        return f"❌ Lỗi khi đọc model: {e}", ""


def on_validate_data(data_path):
    """Callback khi user bấm Validate calibration data."""
    if not data_path or not data_path.strip():
        return "⚠️ Vui lòng nhập đường dẫn calibration data"

    if _current_model_info["data"] is None:
        return "⚠️ Vui lòng upload model ONNX trước (Tab 1)"

    data_path = data_path.strip()
    model_inputs = _current_model_info["data"]["inputs"]

    try:
        result = validate_calibration_data(data_path, model_inputs)
        formatted = format_validation_result(result)
        logger.info(f"Validation result: {result['message']}")
        return formatted
    except Exception as e:
        logger.exception("Error validating data")
        return f"❌ Lỗi khi validate: {e}"


def on_start_quantize(
    quantize_mode,
    calibration_method,
    output_path,
    calib_data_path,
    progress=gr.Progress(),
):
    """Callback khi user bấm Start Quantization."""
    if _current_model_info["data"] is None:
        return "⚠️ Vui lòng upload model ONNX trước (Tab 1)"

    if not calib_data_path or not calib_data_path.strip():
        return "⚠️ Vui lòng nhập đường dẫn calibration data"

    # Lấy đường dẫn ONNX gốc từ file đã upload
    # (Gradio lưu file tạm, cần lấy đường dẫn thực)
    onnx_path = None
    if _current_model_info["data"]:
        # Tìm lại đường dẫn từ state
        onnx_path = _current_model_info.get("onnx_path")

    if not onnx_path or not os.path.exists(onnx_path):
        return "⚠️ Không tìm thấy file ONNX. Vui lòng upload lại ở Tab 1."

    if not output_path.strip():
        output_path = "outputs/model_quantized.onnx"

    config = QuantConfig(
        onnx_path=onnx_path,
        output_path=output_path.strip(),
        quantize_mode=quantize_mode,
        calibration_method=calibration_method,
        calibration_data_path=calib_data_path.strip(),
    )

    def progress_callback(prog, msg):
        progress(prog, desc=msg)

    result = quantize_model(config, progress_callback=progress_callback)

    return result["message"]


def on_upload_for_quantize(file):
    """Lưu lại đường dẫn ONNX khi upload để dùng cho quantize."""
    if file is not None:
        _current_model_info["onnx_path"] = file.name
    return on_upload_model(file)


def on_check_modelopt():
    """Kiểm tra nvidia-modelopt có sẵn không."""
    available, msg = check_modelopt_available()
    return msg


def create_ui() -> gr.Blocks:
    """Tạo Gradio Blocks UI chính."""

    # Custom CSS
    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .status-bar {
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.9em;
    }
    """

    with gr.Blocks(
        title="QuantTool – ONNX Model Optimizer",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
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
                gr.Markdown("### Upload file ONNX để phân tích cấu trúc model")

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_file = gr.File(
                            label="Upload ONNX Model",
                            file_types=[".onnx"],
                            type="filepath",
                        )

                    with gr.Column(scale=2):
                        model_info_display = gr.Markdown(
                            value="*Chưa có model nào được upload...*",
                            label="Model Information",
                        )

                calib_req_display = gr.Markdown(
                    value="",
                    label="Calibration Requirements",
                    visible=True,
                )

            # ═══════════════════════════════════════════════════════════
            # TAB 2: Calibration Data
            # ═══════════════════════════════════════════════════════════
            with gr.TabItem("📊 Calibration Data", id="tab_calib"):
                gr.Markdown("### Validate calibration data so với model input requirements")

                with gr.Row():
                    with gr.Column(scale=2):
                        calib_path_input = gr.Textbox(
                            label="Đường dẫn Calibration Data",
                            placeholder="/path/to/calib_data.npy hoặc .npz",
                            info="Hỗ trợ: .npy (single array) hoặc .npz (multiple arrays)",
                        )
                    with gr.Column(scale=1):
                        validate_btn = gr.Button(
                            "🔍 Validate",
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
                        quant_mode_dropdown = gr.Dropdown(
                            choices=["int8", "fp8", "int4"],
                            value="int8",
                            label="Quantize Mode",
                            info="int8: cân bằng giữa accuracy và speed. fp8: cho GPU Hopper+. int4: nén mạnh nhất.",
                        )

                        calib_method_dropdown = gr.Dropdown(
                            choices=["minmax", "entropy", "max"],
                            value="minmax",
                            label="Calibration Method",
                            info="minmax: nhanh, phổ biến. entropy: chính xác hơn. max: tối ưu cho TensorRT.",
                        )

                    with gr.Column():
                        quant_calib_path = gr.Textbox(
                            label="Đường dẫn Calibration Data",
                            placeholder="/path/to/calib_data.npy",
                            info="Dùng data đã validate ở Tab 2",
                        )

                        quant_output_path = gr.Textbox(
                            label="Đường dẫn Output",
                            value="outputs/model_quantized.onnx",
                            info="Model đã quantize sẽ được lưu tại đây",
                        )

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
        # Event handlers
        # ═══════════════════════════════════════════════════════════

        # Tab 1: Upload model
        upload_file.change(
            fn=on_upload_for_quantize,
            inputs=[upload_file],
            outputs=[model_info_display, calib_req_display],
        )

        # Tab 2: Validate data
        validate_btn.click(
            fn=on_validate_data,
            inputs=[calib_path_input],
            outputs=[validation_result],
        )

        # Tab 3: Check modelopt
        check_btn.click(
            fn=on_check_modelopt,
            inputs=[],
            outputs=[modelopt_status],
        )

        # Tab 3: Start quantization
        quantize_btn.click(
            fn=on_start_quantize,
            inputs=[
                quant_mode_dropdown,
                calib_method_dropdown,
                quant_output_path,
                quant_calib_path,
            ],
            outputs=[quant_result],
        )

    return app
