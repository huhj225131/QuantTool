"""
QuantTool – ONNX Model Optimization Tool
=========================================

Entry point chính. Khởi chạy Gradio UI.

Usage:
    python app.py
    python app.py --port 7860 --share
"""

import argparse

from ui.gradio_app import create_ui
from src.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="QuantTool – ONNX Model Optimizer")
    parser.add_argument("--port", type=int, default=7860, help="Port cho Gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--share", action="store_true", help="Tạo public link qua Gradio")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("QuantTool v0.1.0 – Starting...")
    logger.info("=" * 60)

    app = create_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
