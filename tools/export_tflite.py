#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Dict
from loguru import logger

import numpy as np

from yolox.exp import get_exp

import tensorflow as tf


def make_parser():
    parser = argparse.ArgumentParser("YOLOX tflite deploy")
    parser.add_argument(
        "--onnx-path", type=str, default="yolox.onnx", help="input path of onnx model"
    )
    parser.add_argument(
        "--output-name", type=str, default="yolox.tflite", help="output name of models"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="enable mixed precision quantization for int8 weights and fp16/fp32 activations",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="enable full integer quantization for int8 inference",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="enable full integer quantization for fp16 inference",
    )
    parser.add_argument(
        "--per_tensor",
        action="store_true",
        help="enable per-tensor quantization for int8 inference (disable per-channel)",
    )
    parser.add_argument(
        "--full_int8",
        action="store_true",
        help="enable int8 inputs/outputs for int8 inference (NPU/TPU compatible)",
    )

    return parser


def freeze_graph(model_dir: Path) -> Path:
    """
    Converts saved_model into a frozen graph.
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    logger.info(f"Freezing graph in {model_dir}")
    output_path = model_dir / "frozen_graph.pb"
    if output_path.exists():  # clean up previous exports
        output_path.unlink()

    model = tf.saved_model.load(str(model_dir))
    infer = model.signatures["serving_default"]

    infer_wrapper = tf.function(lambda input_1: infer(input_1))
    concrete_func = infer_wrapper.get_concrete_function(
        tf.TensorSpec(infer.inputs[0].shape, infer.inputs[0].dtype)
    )
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=str(output_path.parent),
        name=str(output_path.name),
        as_text=False,
    )

    if output_path.exists():
        logger.info(f"Graph frozen to {output_path}")
    else:
        raise FileNotFoundError(f"Could not freeze graph to {output_path}")
    return output_path


def load_frozen_graph(
    model_buffer: bytes, inputs: Dict[str, str], outputs: Dict[str, str]
) -> Callable:
    """
    Loads frozen graph into concrete function.
    Uses inputs and outputs to prune the graph and create a signature.
    """
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(model_buffer)

    def _imports_graph_def():
        """
        Wrapper TF2 function.
        """
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    # Prune the graph
    input_tensors = tf.nest.map_structure(import_graph.as_graph_element, inputs)
    output_tensors = tf.nest.map_structure(import_graph.as_graph_element, outputs)
    infer = wrapped_import.prune(feeds=input_tensors, fetches=output_tensors)

    return infer


def export_onnx2tf(onnx_model_path: Path) -> Path:
    """
    Exports ONNX model to TensorFlow using onnx2tf.
    """
    import onnx2tf

    logger.info(f"Exporting {onnx_model_path} to TensorFlow using onnx2tf")
    output_dir = onnx_model_path.parent / "tensorflow"
    if output_dir.exists():  # clean up previous exports
        shutil.rmtree(output_dir)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_model_path),
        output_folder_path=str(output_dir),
        copy_onnx_input_output_names_to_tflite=True,  # preserve input/output names
        output_signaturedefs=True,  # add default signature def
        verbosity="info",
        # disable_group_convolution=True,
    )
    if len(list(output_dir.glob("*.pb"))) > 0:
        logger.info(f"Model exported to {output_dir}")
    else:
        raise FileNotFoundError(f"Could not export model to {output_dir}")
    frozen_graph_path = freeze_graph(output_dir)

    return frozen_graph_path


def export_tflite(
    saved_model_path: Path,
    exp: Any,
    output_name: str = "yolox.tflite",
    quantization: str = "no",
) -> Path:
    """
    Exports given TF saved_model to tflite.
    """
    logger.info(f"Exporting {saved_model_path} to TFLite")
    tflite_path = saved_model_path / f"{output_name}"
    if tflite_path.exists():  # clean up previous exports
        tflite_path.unlink()

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_path), signature_keys=["serving_default"]
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,  # TensorFlow ops
    ]
    if quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if "int8" in quantization:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # override
        dataset = exp.get_dataset()

        def representative_dataset():
            for i in range(min(len(dataset), 1000)):  # 1000 is the max samples
                img = dataset.read_img(i)  # 0-255 uint8 HWC
                padded_image = (
                    np.ones((exp.input_size[0], exp.input_size[1], 3), np.uint8) * 114
                )
                padded_image[: img.shape[0], : img.shape[1]] = img
                input_tensor = padded_image.astype(np.float32)[None, ...]
                yield [input_tensor]

        converter.representative_dataset = representative_dataset
    elif "fp16" in quantization:
        converter.target_spec.supported_types = [tf.float16]

    if "per_tensor" in quantization:
        # Turn off per-channel quantization to support NPUs
        converter._experimental_disable_per_channel = True
    if "full_int8" in quantization:
        # Inputs/outputs as uint8 for NPUs/TPUs compatibility
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as tflite_file:
        tflite_file.write(tflite_model)
    if tflite_path.exists():
        logger.info(f"Model exported to {tflite_path}")
    else:
        raise FileNotFoundError(f"Could not export model to {tflite_path}")

    return tflite_path


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, None)

    quantization = ""
    if args.int8:
        quantization += "_int8"
    if args.full_int8:
        quantization += "_full_int8"
        args.int8 = True
    if args.fp16:
        if args.int8:
            raise ValueError("FP16 and INT8 quantization cannot be enabled at the same time.")
        quantization += "_fp16"
    if args.mixed:
        if args.int8 or args.fp16:
            raise ValueError(
                "Mixed precision quantization cannot be enabled with INT8/FP16 quantization."
            )
        quantization += "_mixed"
    if args.per_tensor:
        if not args.int8:
            raise ValueError("Per-tensor quantization can only be enabled with INT8 quantization.")
        quantization += "_per_tensor"

    frozen_graph_path = export_onnx2tf(Path(args.onnx_path))
    _ = export_tflite(frozen_graph_path.parent, exp, args.output_name, quantization)


if __name__ == "__main__":
    main()
