import time

import cv2
import copy
import numpy as np

from .predict_rec import TextRecognizer
from .predict_det import TextDetector
from .predict_cls import TextClassifier
from .config import Config


class PredictSystem:
    def __init__(self, **kwargs):
        self.args = Config(**kwargs)
        self.text_detector = TextDetector(self.args)
        self.text_recognizer = TextRecognizer(self.args)
        self.use_angle_cls = self.args.use_angle_cls
        self.drop_score = self.args.drop_score
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(self.args)

    def __call__(self, img: np.ndarray) -> (list[np.ndarray], list[tuple], dict):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        ori_im = img.copy()
        start = time.time()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        if dt_boxes is None:
            # logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            pass
            # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        img_crop_list = []
        dt_boxes = sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict['cls'] = elapse
            # logger.debug("cls num  : {}, elapse : {}".format(len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        # logger.debug("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    @staticmethod
    def get_rotate_crop_image(img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    _sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(_sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes
