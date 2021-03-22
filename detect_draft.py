import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
##########
from datasets_model import LoadStreams, MyLoadImages
##########
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# def detect(save_img=False):
#     '''获取输出文件夹，输入源，权重，参数与等信息'''
#     source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
#         ('rtsp://', 'rtmp://', 'http://'))
#
#     # Directories
#     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run #
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 增加运行参数，原来的参数是通过命令行解析对象提供的，这里改为由调用者在代码中提供。需要一个
    # 大体上完成一样功能的参数对象。
    # 我想要的功能是传一组由cv2读取的图片，交给api，然后得到一组打上标签的图片，以及每张图片对应的标签类别引索，位置信息，置信度的信息，还有类别名称字典
    # 要实现这个功能，需要权重文件，输入文件两个参数，其他参数与原代码命令行默认参数保持一致就行。
class simulation_opt:  # 参数对象。

    def __init__(self, weights, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False,
                     classes=None, agnostic_nms=False, augment=False, update=False, exist_ok=False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok

# 增加一个新类，这个新类是在原来detect函数上进行删减。可以先复制原来的detect函数代码，再着手修改
class detectapi:
    def __init__(self, weights, img_size=640):
        # 构造函数中先做好必要的准备，如初始化参数，加载模型
        ''' 删掉
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
        '''  # 改为
        self.opt = simulation_opt(weights=weights, img_size=img_size)
        weights, imgsz = self.opt.weights, self.opt.img_size

        ''' 删掉
        # Directories
        #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        '''

    # Initialize
        set_logging()
        self.device = select_device(self.opt.device)  # 获取设备
        # 如果设备为GPU 使用float16
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # 加载float32模型，确保用户设定的输入图片分辨率能整除32（如不能则调整为能整除返回）
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            # 设置float16
            self.model.half()  # to FP16

    # Second-stage classifier
    # 设置第二次分类，默认不使用
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
            '''
            self.names,和self.colors是由后面的代码拉到这里来的。names是类别名称字典，colors是画框时用到的颜色。
            '''
            # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    def detect(self,source): # 使用时，调用这个函数
        if type(source)!=list:
                raise TypeError('source must be a list which contain  pictures read by cv2')


        vid_path, vid_writer = None, None

        """删掉
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            save_img = True
            # 如果哦检测视频的时候想显示出来，可以在这里加一行 view_img = True
            # view_img = True
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        """
            # 改为
            # Set Dataloader
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)
            # 原来是通过路径加载数据集的，现在source里面就是加载好的图片，所以数据集对象的实现要
            # 重写。修改代码后附。在utils.dataset.py上修改

        '''移动到构造方法末尾。names是类别名称字典，colors是画框时用到的颜色。
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        '''

    # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        result=[]
        ''' 删掉
        for path, img, im0s, vid_cap in dataset: 因为不用保存，所以path可以不要，因为不处理视频，所以vid_cap不要。
        ''' #改为
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized() #计算预测用时的，可以不要
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized() #计算预测用时的，可以不要

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.model, img, im0s)

        # for path, img, im0s, vid_cap in dataset:
        #     img = torch.from_numpy(img).to(device)
        #     # 图片也设置为Float16
        #     img = img.half() if half else img.float()  # uint8 to fp16/32
        #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #     # 没有batch_size的话则在最前面添加一个轴
        #     if img.ndimension() == 3:
        #         img = img.unsqueeze(0)
        # Inference
        # t1 = time_synchronized()
        # pred = model(img, augment=opt.augment)[0]

        # Apply NMS
            # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # t2 = time_synchronized()

        # Apply Classifier
        # 添加二次分类，默认不使用
        #     if self.classify:
        #         pred = apply_classifier(pred, modelc, img, im0s)

        '''删掉
        # Process detections
        # 对每一张图片作处理
        for i, det in enumerate(pred):  # detections per image
        # 如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
        # 设置保存图片/视频的路径
            save_path = str(save_dir / p.name)  # img.jpg
        # 设置保存框坐标txt文件的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
                    if save_txt:  # Write to file
                    # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            '''
            # 改为
            # Process detections
        det=pred[0] #原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
        im0 = im0s.copy() # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        result_txt = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
        if len(det):
                # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
            '''
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            '''
                # Write results
            for *xyxy, conf, cls in reversed(det):

                #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                result_txt.append(line)
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
            result.append((im0,result_txt)) # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        return result, self.names


        # Print time (inference + NMS)
        # 打印前向传播+nms时间
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        # 如果设置展示，则show图片/视频
        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond
        #
        # # Save results (image with detections)
        # # 设置保存图片/视频
        # if save_img:
        #     if dataset.mode == 'image':
        #         cv2.imwrite(save_path, im0)
        #     else:  # 'video'
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer
        #
        #             fourcc = 'mp4v'  # output video codec
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        #         vid_writer.write(im0)
        #
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")
        #     # 打开保存图片和txt的路径(好像只适用于MacOS系统)
        # # 打印总时间
        # print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否也去除不同类别之间的框，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
        """
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
    print(opt)
    check_requirements()

weights = 'weights/yolov5s.pt'
detect = detectapi(weights)  #######
with torch.no_grad():
    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect()
            strip_optimizer(opt.weights)
    else:
        detect()
