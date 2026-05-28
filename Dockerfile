FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /

# Install build dependencies first (needed for pynini compilation)
RUN pip install Cython wheel setuptools

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY rp_handler.py /

# Pre-download IndexTTS-2 model weights for faster cold start
COPY prefetch_models.py /prefetch_models.py
RUN python /prefetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
