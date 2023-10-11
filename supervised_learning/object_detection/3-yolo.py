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
            self.class_names = [line for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
        Each output will have the shape
        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
        grid_height & grid_width => the height and width of the grid used for
        the output
        anchor_boxes => the number of anchor boxes used
        4 => (t_x, t_y, t_w, t_h)
        1 => box_confidence
        classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the processed
        boundary boxes for each output, respectively:
        4 => (x1, y1, x2, y2)
        (x1, y1, x2, y2) should represent the boundary box relative to
        original image
        box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1) containing the box
        confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes) containing the box’s
        class probabilities for each output, respectively
        """
        boxes, box_confidences, box_class_probs = [], [], []
        image_height, image_width = image_size
        anchors_w = self.anchors[..., 0]
        anchors_h = self.anchors[..., 1]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchors_size, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]
            box_confidence = output[..., 4]
            classes = output[..., 5:]

            p_h = np.tile(anchors_h[i], grid_height).reshape(
                grid_height, 1, len(anchors_h[i]))
            p_w = np.tile(anchors_w[i], grid_width).reshape(
                grid_width, 1, len(anchors_w[i]))

            c_x = np.tile(np.arange(grid_width), grid_width).reshape(
                grid_width, grid_width, 1)
            c_y = np.tile(np.arange(grid_height), grid_height).reshape(
                grid_width, grid_width).T.reshape(
                grid_height, grid_height, 1)

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height
            b_w = (np.exp(t_w) * p_w) / self.model.input.shape[1].value
            b_h = (np.exp(t_h) * p_h) / self.model.input.shape[2].value

            box = np.empty((grid_height, grid_width, anchors_size, 4))

            box[..., 0] = (b_x - b_w / 2) * image_width
            box[..., 1] = (b_y - b_h / 2) * image_height
            box[..., 2] = (b_x + b_w / 2) * image_width
            box[..., 3] = (b_y + b_h / 2) * image_height

            boxes.append(box)

            box_confidences.append((1 / (1 + np.exp(-box_confidence))
                                    ).reshape(grid_width, grid_height,
                                              anchors_size, 1))

            box_class_probs.append(1 / (1 + np.exp(-classes)))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """boxes: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 4) containing the processed
        boundary boxes for each output, respectively
        box_confidences: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, 1) containing the processed box
        confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape
        (grid_height, grid_width, anchor_boxes, classes) containing the
        processed box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        that each box in filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        """
        scores, filtered_boxes, classes = [], [], []
        for i, box in enumerate(boxes):

            box_scores = box_confidences[i] * box_class_probs[i]

            box_classes = np.argmax(box_scores, -1)

            box_classes_scores = np.max(box_scores, -1)

            filter_mask = box_classes_scores > self.class_t

            scores.extend(box_classes_scores[filter_mask].tolist())
            filtered_boxes.extend(box[filter_mask].tolist())
            classes.extend(box_classes[filter_mask].tolist())

        return (np.array(filtered_boxes), np.array(classes), np.array(scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
        the filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores):
        box_predictions: a numpy.ndarray of shape (?, 4) containing all of the
        predicted bounding boxes ordered by class and box score
        predicted_box_classes: a numpy.ndarray of shape (?,) containing the
        class number for box_predictions ordered by class and box score,
        respectively
        predicted_box_scores: a numpy.ndarray of shape (?) containing the box
        scores for box_predictions ordered by class and box score, respectively
        """

        def intersection_over_union(box_1, box_2):
            """Finds the intersection over union between two boxes
            """
            xi_1 = max(box_1[0], box_2[0])
            yi_1 = max(box_1[1], box_2[1])
            xi_2 = min(box_1[2], box_2[2])
            yi_2 = min(box_1[3], box_2[3])

            intersection = max(0, yi_2 - yi_1 + 1) * max(0, xi_2 - xi_1 + 1)

            box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
            box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

            union = box_1_area + box_2_area - intersection

            return intersection / union

        class_order = np.argsort(box_classes)
        filtered_boxes = filtered_boxes[class_order]
        box_classes = box_classes[class_order]
        box_scores = box_scores[class_order]

        separator_indices = np.where(box_classes[:-1] != box_classes[1:])[0]
        box_scores = np.split(box_scores, separator_indices + 1)
        filtered_boxes = np.split(filtered_boxes, separator_indices + 1)

        scores_idxs = [np.argsort(box_score)[::-1] for box_score in box_scores]
        filtered_boxes = [filtered_box[scores_idxs[i]
                                       ] for i, filtered_box in enumerate(
                                           filtered_boxes)]

        box_scores = [np.sort(box_score)[::-1] for box_score in box_scores]

        best_filtered_boxes = [
            filtered_box[0] for filtered_box in filtered_boxes]

        box_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        better_iou_boxes = []
        for i, filtered_boxes_class in enumerate(filtered_boxes):
            for j, filtered_box in enumerate(filtered_boxes_class):
                if better_iou_boxes:
                    for better_iou_box in better_iou_boxes:
                        iou = intersection_over_union(better_iou_box,
                                                      filtered_box)
                else:
                    iou = intersection_over_union(best_filtered_boxes[i],
                                                  filtered_box)
                if iou <= self.nms_t or np.array_equal(filtered_box,
                                                       best_filtered_boxes[i]):
                    box_predictions.append(filtered_box)
                    predicted_box_classes.append(box_classes[i])
                    predicted_box_scores.append(box_scores[i][j])
                    better_iou_boxes.append(filtered_box)
            better_iou_boxes = []

        return np.array(box_predictions), np.array(
            predicted_box_classes), np.array(predicted_box_scores)
