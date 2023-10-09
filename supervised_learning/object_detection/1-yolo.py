#!/usr/bin/env python3
"""Yolo Class Module"""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np


class Yolo():
    """class Yolo that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            data = f.read()
            self.class_names = data.split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        box, boxes, box_confidences, box_class_probs = [], [], [], []
        image_height, image_width = image_size
        print(image_height, image_width)
        for output in outputs:
            print(output.shape)
            # box.append(output[:, :, :, 0] / image_width)
            # box.append(output[:, :, :, 1] / image_height)
            # box.append(output[:, :, :, 2] / image_width)
            # box.append(output[:, :, :, 3] / image_height)
            boxes.append(
                (output[..., 0] * 2 + output[..., 2]) / (image_width * 2))
            box_confidences.append(output[:, :, :, 4])
            box_class_probs.append(output[:, :, :, 5:])
        return (boxes, box_confidences, box_class_probs)
