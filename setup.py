from setuptools import setup, find_packages

setup(
    name="quant_tool",
    version="0.1.0",
    description="ONNX Model Optimization Tool using NVIDIA ModelOpt with Gradio UI",
    author="Jeson",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gradio>=4.0",
        "onnx>=1.14",
        "onnxruntime>=1.16",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "tqdm>=4.65",
        "nvidia-modelopt[all]",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "quant-tool=app:main",
        ],
    },
)
