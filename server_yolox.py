import os
import json
import time
import logging
import argparse
import cv2 
from flask import Flask, Response, request, render_template, send_from_directory

from loguru import logger

import cv2

from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis


# from tracker.multitracker import JDETracker
# from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
import utils.datasets as datasets
import torch
from utils.utils import *


from flask_cors import CORS
from flask_cors import cross_origin
traker=None
cam,frame_rate=None,None
bool_video=None
w,h=None,None

public_cls = list() #裝目前偵測到的cls類別
select_cls_list = list() #裝目前選定的cls類別

app = Flask(__name__, template_folder='./')
CORS(app)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print('Release camera')
    cam.release()


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
    # print("in letterbox")
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def get_size(vw, vh, dw, dh):
    wa, ha = float(dw) / vw, float(dh) / vh
    a = min(wa, ha)
    return int(vw *a), int(vh*a)

def eval_seq():
    global public_cls
    frame_id = 0

    while True:
        _,img0=cam.read()
        


        if(bool_video):
            img0 = cv2.resize(img0, (w, h))

        # if frame_id %2 != 0:
        if frame_id %2 < -1:
            print("in first if/else")
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id)
            ret, jpeg = cv2.imencode('.jpg', online_im)
            frame = jpeg.tobytes()
        else:
            img, _, _, _ = letterbox(img0, width=w, height=h)
            outputs, img_info = predictor.inference(img)
            
            # outputs = outputs.cpu()
            output = outputs[0].cpu()
            bboxes = output[:, 0:4] / img_info["ratio"]
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            # online_tlwhs = list(bboxes[scores > 0.75])
            # online_ids = list(cls[scores > 0.75])

            # save results
            # results.append((frame_id + 1, online_tlwhs, online_ids))
            
            
            # online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id)
            # cls:是此frame所含的object class
            online_im, detect_cls_list = vis(img, bboxes, scores, cls, 0.75, COCO_CLASSES, select_cls_list)
            # print(f"type: {type(cls)}")
            public_cls = detect_cls_list.copy()
            # print(f"now public_cls:{public_cls}")
            ret, jpeg = cv2 .imencode('.jpg', online_im)
            frame =  jpeg.tobytes()

        frame_id += 1
        
           

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n'
        b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n'
        b'\r\n' + frame + b'\r\n')
        # print(b'--frame\r\n'
        # b'Content-Type: image/jpeg\r\n'
        # b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n')

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response



@app.route('/control_cls')
@cross_origin()
def control_cls():
    select_cls = request.args.get('select_cls')
    # act: select -> 選取, deselect -> 取消
    print("select_cls: ",select_cls)
    act = request.args.get('act')
    print("act: ",act)
    if act == "select":
        
        cls_id = COCO_CLASSES.index(select_cls)
        print("select_cls_id ",cls_id)
        select_cls_list.append(cls_id)
        print("select_cls_list: ",select_cls_list)
    else:

        cls_id = COCO_CLASSES.index(select_cls)
        print("diselect_cls_id ",cls_id)
        if(select_cls_list and cls_id in select_cls_list):
            select_cls_list.remove(cls_id)
        print("select_cls_list: ",select_cls_list)
    return json.dumps({'success':True, 'target':select_cls}), 200, {'ContentType': 'appliccation/json'}

@app.route('/video_feed')
def video_feed():
    # 回傳影片資訊
    return Response(eval_seq(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_cls')
def get_cls():
    global public_cls
    # public_cls= ["1","2"]
    return json.dumps({'success':True, 'target':public_cls}), 200, {'ContentType': 'appliccation/json'}

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Flask server shutting down...'

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo_choose.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    # parser.add_argument('--weights', type=str, default='jde.1088x608.uncertainty.pt', help='path to weights file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str,default="webcam0",help='path to the input video')
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # exp file
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(640, 640), type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="",
        type=str,
        help="please input your experiment description file",
    )


    opt = parser.parse_args()
    print(opt, end='\n\n')


    # run tracking
    #eval_seq(opt)
        

    w, h = opt.tsize

    if(opt.input_video=="webcam0"):
        print("enter webcam0 >> setting webcam0")
        cam=cv2.VideoCapture(0)
        bool_video=False
        frame_rate = int(round(cam.get(cv2.CAP_PROP_FPS)))
        
    else:
        if not os.path.isfile(opt.input_video):
            raise FileExistsError
        bool_video=True
        cam = cv2.VideoCapture(opt.input_video)
        frame_rate = int(round(cam.get(cv2.CAP_PROP_FPS)))
        
        vw = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        imwidth = opt.tsize[0]
        imheight = opt.tsize[1]
        
        w, h = get_size(vw, vh, imwidth, imheight)
    
    # tracker = JDETracker(opt, frame_rate)
    
    exp = get_exp(opt.exp_file, opt.name)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if opt.device == "gpu":
        model.cuda()
        if opt.fp16:
            model.half()  # to FP16
    model.eval()

    if not opt.trt:
        if opt.ckpt is None:
            # ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            print("you must set ckpt")
        else:
            ckpt_file = opt.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if opt.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        opt.device, opt.fp16, opt.legacy,
    )
    current_time = time.localtime()

    config = json.load(open('config.json', 'r'))
    app.run(host='0.0.0.0', port=config['flask_port'], debug=True)