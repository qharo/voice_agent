FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    pkg-config \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswresample-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libffi-dev \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models && \
    wget -O /models/en_US-lessac-high.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx && \
    wget -O /models/en_US-lessac-high.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json && \
    wget -O /models/en_US-amy-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx && \
    wget -O /models/en_US-amy-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

RUN wget -O /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz && \
    tar -xzf /tmp/piper.tar.gz -C /usr/local && \
    rm /tmp/piper.tar.gz && \
    ln -s /usr/local/piper/piper /usr/local/bin/piper

ENV LD_LIBRARY_PATH="/usr/local/piper:$LD_LIBRARY_PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV FASTEMBED_CACHE_PATH=/models/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN wget -O /models/moonshine-streaming-tiny-Q8_0.gguf \
    https://huggingface.co/handy-computer/moonshine-streaming-tiny-gguf/resolve/main/moonshine-streaming-tiny-Q8_0.gguf

# Pocket TTS 100M (soniqo ONNX INT8, fixed Alba voice) - streaming TTS engine
RUN mkdir -p /models/pocket-tts && \
    for f in config.json decoder.int8.onnx encoder.onnx lm_flow.int8.onnx \
             lm_main.int8.onnx text_conditioner.onnx token_scores.json \
             tokenizer.model vocab.json; do \
        wget -O /models/pocket-tts/$f \
            "https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8/resolve/main/$f"; \
    done

RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('owensong/Inflect-Nano-v2-ONNX', local_dir='/models/Inflect-Micro-v2-ONNX')" && \
    pip install --no-cache-dir -r /models/Inflect-Micro-v2-ONNX/onnx/requirements.txt && \
    rm -rf /root/.cache/huggingface

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
