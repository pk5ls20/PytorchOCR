class Config:
    def __init__(self, **kwargs):
        # prediction engine parameters
        self.use_gpu = True
        self.gpu_mem = 500
        self.warmup = False
        # text detector parameters
        self.image_dir = None
        self.det_algorithm = 'DB'
        self.det_model_path = None
        self.det_limit_side_len = 960
        self.det_limit_type = 'max'
        # DB parameters
        self.det_db_thresh = 0.3
        self.det_db_box_thresh = 0.6
        self.det_db_unclip_ratio = 1.5
        self.max_batch_size = 10
        self.use_dilation = False
        self.det_db_score_mode = "fast"
        # EAST parameters
        self.det_east_score_thresh = 0.8
        self.det_east_cover_thresh = 0.1
        self.det_east_nms_thresh = 0.2
        # SAST parameters
        self.det_sast_score_thresh = 0.5
        self.det_sast_nms_thresh = 0.2
        self.det_sast_polygon = False
        # PSE parameters
        self.det_pse_thresh = 0
        self.det_pse_box_thresh = 0.85
        self.det_pse_min_area = 16
        self.det_pse_box_type = 'box'
        self.det_pse_scale = 1
        # FCE parameters
        self.scales = [8, 16, 32]
        self.alpha = 1.0
        self.beta = 1.0
        self.fourier_degree = 5
        self.det_fce_box_type = 'poly'
        # recognizer parameters
        self.rec_algorithm = 'CRNN'
        self.rec_model_path = None
        self.rec_image_inverse = True
        self.rec_image_shape = "3, 48, 320"
        self.rec_char_type = 'ch'
        self.rec_batch_num = 6
        self.max_text_length = 25
        self.use_space_char = True
        self.drop_score = 0.5
        self.limited_max_width = 1280
        self.limited_min_width = 16
        # classifier parameters
        self.use_angle_cls = False
        self.cls_model_path = None
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        # e2e parameters
        self.e2e_algorithm = 'PGNet'
        self.e2e_model_path = None
        self.e2e_limit_side_len = 768
        self.e2e_limit_type = 'max'
        self.e2e_pgnet_score_thresh = 0.5
        self.e2e_pgnet_valid_set = 'totaltext'
        self.e2e_pgnet_polygon = True
        self.e2e_pgnet_mode = 'fast'
        # SR parameters
        self.sr_model_path = None
        self.sr_image_shape = "3, 32, 128"
        self.sr_batch_num = 1
        # YAML paths
        self.det_yaml_path = None
        self.rec_yaml_path = None
        self.cls_yaml_path = None
        self.e2e_yaml_path = None
        self.sr_yaml_path = None
        # multi-process parameters
        self.use_mp = False
        self.total_process_num = 1
        self.process_id = 0
        # other parameters
        self.benchmark = False
        self.save_log_path = "./log_output/"
        self.show_log = True
        # font paths
        self.script_dir = None   # os.path.dirname(os.path.abspath(__file__))
        self.vis_font_path = None  # doc/fonts/simfang.ttf'
        self.rec_char_dict_path = None  # 'pytorchocr/utils/ppocr_keys_v1.txt'
        self.e2e_char_dict_path = None  # 'pytorchocr/utils/ic15_dict.txt'
        self.enable_mkldnn = False
        self.use_pdserving = False

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
