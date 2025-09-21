#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import demo_postprocess, mkdir, multiclass_nms, vis

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter  # noqa: N806


def make_parser():
    parser = argparse.ArgumentParser("tflite inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.tflite",
        help="Input your tflite model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # Load model
    model_buffer = open(args.model, "rb").read()
    loaded = Interpreter(model_content=model_buffer, num_threads=4)
    infer = loaded.get_signature_runner("serving_default")

    # Read model input-output details
    input_details = infer.get_input_details()
    input_arg = list(input_details.keys())[0]
    input_shape = input_details[input_arg]["shape"]
    input_size = (input_shape[2], input_shape[1])
    input_dtype = input_details[input_arg]["dtype"]
    output_details = infer.get_output_details()
    output_arg = list(output_details.keys())[0]

    # Preprocess the input image
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape[1:], swap=(0, 1, 2))
    batch_image = img.astype(input_dtype)[None, ...]

    # Perform inference and postprocessing
    output = infer(**{"images": batch_image})
    predictions = demo_postprocess(output[output_arg], input_shape[1:])[0]

    # Rescale outputs to original image size
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio

    # NMS and visualization
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=COCO_CLASSES)

    # Save the results
    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
