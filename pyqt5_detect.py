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

import sys
import clr
#C#库的地址
#clr.AddReference(r'C:\Users\64504\Desktop\ptzcontrol.csharp\ConsoleApplication1\bin\Debug\SDK.IPC.dll')
#from SDK.IPC import  *
#from System import  *
#from System.Threading import *


import argparse
import time
from pathlib import Path


cam_ip = "183.192.69.170"
camBG_port = "7502"
camCS_port = "7702"
cam_port = camCS_port
cam_user = "admin"
cam_psw = "SMUwm_007"

weights= r'./ship-lin.pt'

class pyqt_detect_api:
    def __init__(self,vWidth,vHeight):


        # 由于视频像素是1920*1080 因此对应qt界面1200*900的比例是1.6
        self.scale = 1.6  # 监控像素缩放到ui像素比例
        # 相机中心点
        self.midX = (vWidth * self.scale) / 2
        self.midY = (vHeight * self.scale) / 2

        self.cam_speed = 1  # 相机转动速度
        # 检测框坐标
        self.trackX = 0
        self.trackY = 0
        self.predList = []  # 保存每一帧检测到的坐标
        self.TRACKFLAG = False
        # 上一帧坐标
        self.tracked_X = 0
        self.tracked_Y = 0

        # C#相机库
        ipc = IPCControl()
        ipcKey = "key11111111111111111111111111111111111"
        # ipcConParam = IPCConnectParam("183.192.69.170", 7701, "admin", "SMUwm_007")
        ipcConParam = IPCConnectParam(cam_ip, int(cam_port) - 1, cam_user, cam_psw)
        self.ipcCon = ipc.GetPtzConnecttedControl(1, ipcConParam, ipcKey)
        # ipcCon.Move(EnumPtzMoveType.moveleft);
        self.ipcCon.Move(EnumPtzMoveType.movestop)  # 让相机初始化时发出一个停止转动的命令

    def pyqt_detect(self,show_frame,tracked_X,tracked_Y):
        self.tracked_X = tracked_X
        self.tracked_Y = tracked_Y
        print("XXXXXXXX",self.tracked_X)
        print("YYYYYYYY",self.tracked_Y)
        source = r'rtsp://' + cam_user + r':' + cam_psw + r'@' + cam_ip + r':' + cam_port + r'/id=1'
        # source=r'rtsp://admin:SMUwm_007@192.168.1.110/id=1'
        # weights= r'./yolov5x.pt'
        view_img = True
        imgsz = 640
        # = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
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
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
            t2 = time_synchronized()

            # Process detections
            # 检测每一帧图片
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
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
                    countCls = 0
                    self.predList.clear()

                    checkTracked = False
                    for *xyxy, conf, cls in reversed(det):
                        # 画框
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                        # 判断tracked是否与上一针存在联系
                        cX = 0
                        cY = 0
                        # print(self.TRACKFLAG)
                        # print(checkTracked)
                        #print(abs(self.tracked_X - cX), abs(self.tracked_Y - cY))
                        print(abs(self.tracked_X - cX), abs(self.tracked_Y - cY))
                        if self.TRACKFLAG:
                            cX = (c1[0] + c2[0]) / 2
                            cY = (c1[1] + c2[1]) / 2
                            # print((c1[0]+c2[0])/2,(c1[1]+c2[1])/2)

                            # 如果追踪到了则按差值移动相机
                            print('loss:', self.tracked_X - self.midX)
                            if (self.tracked_X - self.midX < -10):
                                print('left:', self.tracked_X - self.midX)
                                self.ipcCon.Move(EnumPtzMoveType.moveleft, self.cam_speed)
                                self.ipcCon.Move(EnumPtzMoveType.movestop)
                            elif (self.tracked_X - self.midX > 10):
                                print('right:', self.tracked_X - self.midX)
                                self.ipcCon.Move(EnumPtzMoveType.moveright, self.cam_speed)
                                self.ipcCon.Move(EnumPtzMoveType.movestop)
                            else:

                                self.ipcCon.Move(EnumPtzMoveType.movestop)
                                # self.TRACKFLAG=False

                            if abs(self.tracked_X - cX) < 50 and abs(self.tracked_Y - cY) < 50:
                                self.tracked_X = cX
                                self.tracked_Y = cY
                                checkTracked = True
                                break
                                # print('tracked_X',self.tracked_X)
                                # print('tracked_Y',self.tracked_Y)
                            else:
                                checkTracked = False
                                self.ipcCon.Move(EnumPtzMoveType.movestop)
                        # 存储当前帧的检测框与下一帧关联起来
                        predBox = (c1, c2)
                        self.predList.append(predBox)
                        # print(c1)
                        # print("******************")

                        # print("ffffffffffffffffffffffff")
                        # print(self.predList[-1])
                        # print("ffffffffffffffffffffffff")
                        # print(label,c1,c2)

                    if checkTracked == False:
                        self.TRACKFLAG = False
                    print(self.TRACKFLAG)
                    # print(checkTracked)

                    # for pred in self.predList:

                    # print(pred[:1])
                # Print time (inference + NMS)
                # 打印前向传播+nms时间
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                # 调整显示的像素排列
                cv2.cvtColor(im0, cv2.COLOR_BGR2RGB, im0)
                show_frame(im0)