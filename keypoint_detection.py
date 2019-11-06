# coding=utf-8

# 加载一些基础包以及设置logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# 加载其它一些库
import numpy as np
import cv2

# 加载相关工具
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

if __name__ == '__main__':
    input_path = "input1.jpg"
    output_path = "result_keypoint_detection.jpg"

    # 指定模型的配置配置文件路径及网络参数文件的路径
    # 对于像下面这样写法的网络参数文件路径，程序在运行的时候就自动寻找，如果没有则下载。
    # Keypoint detection model
    model_file_path = "configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    model_weights = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"

    # 加载图片
    img = cv2.imread(input_path)

    # 创建一个detectron2配置
    cfg = get_cfg()
    # 要创建的模型的名称
    cfg.merge_from_file(model_file_path)
    # 为模型设置阈值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # 加载模型需要的数据
    cfg.MODEL.WEIGHTS = model_weights
    # 基于配置创建一个默认推断
    predictor = DefaultPredictor(cfg)
    # 利用这个推断对加载的影像进行分析并得到结果
    # 对于输出结果格式可以参考这里https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    outputs = predictor(img)

    # 控制台中输出一些结果
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # 得到结果后可以使用Visualizer对结果可视化
    # img[:, :, ::-1]表示将BGR波段顺序变成RGB
    # scale表示输出影像的缩放尺度，太小了会导致看不清
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # 获得绘制的影像
    result = v.get_image()[:, :, ::-1]
    # 将影像保存到文件
    cv2.imwrite(output_path, result)
