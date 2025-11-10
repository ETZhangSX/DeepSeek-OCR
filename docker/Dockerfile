FROM harbor.shopeemobile.com/aip/aip-image-hub/aip-prod/projects/100079/vllm:v0.8.5.post1-71049b17d6

COPY . /workspace/DeepSeek-OCR

WORKDIR /workspace

RUN cd /workspace/DeepSeek-OCR && pip3 install -r requirements.txt && pip3 install flash-attn==2.7.3 --no-build-isolation
