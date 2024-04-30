import numpy as np
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from .predict_system import PredictSystem


class EasyPaddleOCR:
    def __init__(self, use_angle_cls=False, devices="auto", needWarmUp=False,
                 warmup_size=(640, 640), drop_score=0.5, **kwargs):
        """
        Initialize EasyPaddleOCR object with specified parameters.
        :param use_angle_cls: Whether to use angle classifier. This is used to detect text that is 180' rotated.
        :param devices: Device to use in pytorch. "auto" for auto-detect, "cpu" for cpu, "cuda" for cuda.
        :param needWarmUp: Whether to warm up the model. This will take some time.
        :param warmup_size: Size of the image to use in warm up. Please specify the **MAX** image size you want to
        inference with in this parameter when possible. This can reduce the usage of VMEM.
        :param kwargs: other parameters to pass to InferSystem. See original PaddleOCR documentation for more details.
        """
        self._modelFileKeys = ["det_model_path", "rec_model_path", "cls_model_path",
                               "det_model_config_path", "rec_model_config_path", "character_dict_path"]
        self._modelFilePaths = {key: kwargs.get(key, None) for key in self._modelFileKeys}
        if devices == "auto":
            self._use_gpu = True if torch.cuda.is_available() else False
        else:
            self._use_gpu = True if devices == "cuda" else False
        logger.info(f"Using device: {devices}")
        self._download_file(self._modelFilePaths)
        self.ocr = PredictSystem(use_angle_cls=use_angle_cls,
                                 det_yaml_path=self._modelFilePaths["det_model_config_path"],
                                 det_model_path=self._modelFilePaths["det_model_path"],
                                 rec_yaml_path=self._modelFilePaths["rec_model_config_path"],
                                 rec_model_path=self._modelFilePaths["rec_model_path"],
                                 cls_model_path=self._modelFilePaths["cls_model_path"],
                                 rec_char_dict_path=self._modelFilePaths["character_dict_path"],
                                 drop_score=drop_score,
                                 use_gpu=self._use_gpu)
        self.needWarmUp = needWarmUp
        self._warm_up(warmup_size) if self.needWarmUp else None

    @staticmethod
    def _download_file(fileDict):
        config_default_dict = {
            "det_model_path": "PaddleOCR2Pytorch/ch_ptocr_v4_det_infer.pth",
            "rec_model_path": "PaddleOCR2Pytorch/ch_ptocr_v4_rec_infer.pth",
            "cls_model_path": "PaddleOCR2Pytorch/ch_ptocr_mobile_v2.0_cls_infer.pth",
            "det_model_config_path": "PaddleOCR2Pytorch/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml",
            "rec_model_config_path": "PaddleOCR2Pytorch/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
            "character_dict_path": "ppocr_keys_v1.txt"
        }
        for key, val in fileDict.items():
            if not val:
                logger.warning(f"Unspecified {key[:-5]}, using default value {config_default_dict[key]}")
                fileDict[key] = hf_hub_download(repo_id="pk5ls20/PaddleModel", filename=config_default_dict[key])
        logger.info(fileDict)

    def _warm_up(self, warmup_size):
        logger.info("Warm up started")
        assert (isinstance(warmup_size, (list, tuple)) and
                len(warmup_size) == 2), "warmup_size must be tuple or list with 2 elems."
        img = np.random.uniform(0, 255, [warmup_size[0], warmup_size[1], 3]).astype(np.uint8)
        for i in range(10):
            _ = self.ocr(img)
        logger.info("Warm up finished")
