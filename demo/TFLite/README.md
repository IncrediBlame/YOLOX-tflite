## YOLOX-TFLite in Python

This doc introduces how to convert your onnx model into tflite, and how to run a tflite demo to verify your convertion.

### Step1: Install TFLite

run the following command to install tflite:
```shell
pip install tensorflow
```

### Step2: Get ONNX models

See [ONNXRuntime demo](../../demo/ONNXRuntime/README.md). Do not set `--decode_in_inference` flag for onnx export.

### Step3: Convert ONNX model to TFLite

First, you should move to <YOLOX_HOME> by:
```shell
cd <YOLOX_HOME>
```

To convert your ONNX model (without quantization), use:

```shell
python3 tools/export_tflite.py --onnx-path /path/to/yolox.onnx --output-name your_yolox.tflite -f exps/your_dir/your_yolox.py
```

To convert your ONNX model with FP16 quantization, use:

```shell
python3 tools/export_tflite.py --onnx-path /path/to/yolox.onnx --output-name your_yolox.tflite -f exps/your_dir/your_yolox.py --fp16
```

To convert your ONNX model with INT8 quantization, keeping inputs-outputs as FP32, use:

```shell
python3 tools/export_tflite.py --onnx-path /path/to/yolox.onnx --output-name your_yolox.tflite -f exps/your_dir/your_yolox.py --int8
```

To convert your ONNX model with FULL INT8 quantization (setting inputs-outputs as UINT8) use:

```shell
python3 tools/export_tflite.py --onnx-path /path/to/yolox.onnx --output-name your_yolox.tflite -f exps/your_dir/your_yolox.py --full_int8
```

To convert your ONNX model with per-tensor INT8 quantization use:

```shell
python3 tools/export_tflite.py --onnx-path /path/to/yolox.onnx --output-name your_yolox.tflite -f exps/your_dir/your_yolox.py --int8 --per_tensor
```

### Step4: TFLite Demo

Step1.
```shell
cd <YOLOX_HOME>
```

Step2.
```shell
python3 -m demo.TFLite.tflite_inference -m <TFLITE_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3
```
Notes:
* -m: your converted tflite model
* -i: input_image
* -s: score threshold for visualization.
