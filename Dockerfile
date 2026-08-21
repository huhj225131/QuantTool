# Base Image dành riêng cho NVIDIA Jetson (JetPack 6 / L4T r36.4.0)
# Đã tích hợp sẵn: CUDA, cuDNN, TensorRT (trtexec), PyTorch, OpenCV cho ARM64
FROM dustynv/l4t-ml:r36.4.0

# Thiết lập Environment Variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Cài đặt các thư viện hệ thống bổ sung (nếu cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thư mục làm việc
WORKDIR /app

# Copy requirements.txt để cache Docker layer
COPY requirements.txt /app/

# Cài đặt các gói Python cần thiết (dustynv image đã có sẵn PyTorch, torchvision, opencv...)
RUN pip3 install --no-cache-dir -r requirements.txt || true

# Copy source code và cấu hình
COPY setup.py /app/
COPY app.py /app/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY ui/ /app/ui/

# Tạo thư mục outputs và logs
RUN mkdir -p /app/outputs /app/logs

# Port mặc định của Gradio UI
EXPOSE 7860

# Lệnh mặc định khi chạy container
CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "7860"]
