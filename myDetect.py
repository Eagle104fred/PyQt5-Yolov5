import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class my_detect_api(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()


    def run_detect(self):
        return self.detect()

    def detect(self):
        '''获取输出文件夹，输入源，权重，参数与等信息'''
        source=r'rtsp://admin:SMUwm_007@192.168.1.110/id=1'
        weights= r'./yolov5s.pt'
        view_img=True
        imgsz=640
            #= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Initialize
        set_logging()
        device = select_device()  # 获取设备
        # 如果设备为GPU 使用float16
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # 加载float32模型，确保用户设定的输入图片分辨率能整除32（如不能则调整为能整除返回）
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            # 设置float16
            model.half()  # to FP16

        # Second-stage classifier
        # 设置第二次分类，默认不使用
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        # 通过不同的输入源来设置不同的数据加载方式
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            # 如果哦检测视频的时候想显示出来，可以在这里加一行 view_img = True
            view_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        # 获取类别名字
        names = model.module.names if hasattr(model, 'module') else model.names
        # 设置画框的颜色
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        # 进行一次前向推理,测试程序是否正常

        """
           path 图片/视频路径
           img 进行resize+pad之后的图片
           img0 原size图片
           cap 当读取图片时为None，读取视频时为视频源
        """
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            # 图片也设置为Float16
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size的话则在最前面添加一个轴
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            """
                   前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
                   h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
                   num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
                   pred[..., 0:4]为预测框坐标
                   预测框坐标为xywh(中心点+宽长)格式
                   pred[..., 4]为objectness置信度
                   pred[..., 5:-1]为分类结果
            """

            # Apply NMS
            """
                   pred:前向传播的输出
                   conf_thres:置信度阈值
                   iou_thres:iou阈值
                   classes:是否只保留特定的类别
                   agnostic:进行nms是否也去除不同类别之间的框
                   经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
                   pred是一个列表list[torch.tensor]，长度为batch_size
                   每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
            """

            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            t2 = time_synchronized()

            # Apply Classifier
            # 添加二次分类，默认不使用
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            # 对每一张图片作处理
            for i, det in enumerate(pred):  # detections per image
            # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path

            # 设置打印信息(图片长宽)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if  view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                # 打印前向传播+nms时间
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                return im0

                # Stream results
                # 如果设置展示，则show图片/视频
                #if view_img:
                #    cv2.imshow(str(p), im0)
                #    cv2.waitKey(1)  # 1 millisecond


        # 打印总时间
        print(f'Done. ({time.time() - t0:.3f}s)')
