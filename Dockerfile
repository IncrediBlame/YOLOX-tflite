# BUILD: docker build -t yolox-tflite .
# RUN: docker run -it --rm --runtime nvidia --gpus all --user `id -u`:`id -g` -v .:/app yolox-tflite bash

FROM python:3.10

# TODO: some packages might be unnessary, clean up later
RUN pip3 install torch==2.4.1 torchvision opencv-python-headless==4.11.0.86 tensorflow==2.15.1 tensorboard==2.15.2 \
        openvino-dev==2024.6.0 nncf==2.15.0 onnx==1.17.0 onnxruntime==1.20.1 onnx-simplifier==0.4.36 \
        pycocotools==2.0.8 pyyaml loguru tqdm thop ninja tabulate psutil \
        onnx2tf==1.26.8 tf-keras==2.15.1 onnx_graphsurgeon==0.5.5 sng4onnx==1.0.4 torch-pruning==1.5.1

RUN mkdir /.cache && chmod 777 /.cache
RUN mkdir /models && chmod 777 /models
WORKDIR /app
