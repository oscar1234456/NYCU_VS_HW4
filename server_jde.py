import os
import json
import time
import logging
import argparse
import cv2 
from flask import Flask, Response, request, render_template, send_from_directory


from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
import utils.datasets as datasets
import torch
from utils.utils import *

traker=None
cam,frame_rate=None,None
bool_video=None
w,h=None,None

app = Flask(__name__, template_folder='./')


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print('Release camera')
    cam.release()


def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
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
    
    frame_id = 0

    while True:
        ok,img0=cam.read()
        if(ok):
            if(bool_video):
                img0 = cv2.resize(img0, (w, h))

            img, _, _, _ = letterbox(img0)
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            # run tracking
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            # results.append((frame_id + 1, online_tlwhs, online_ids))
            
            
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id)
            ret, jpeg = cv2.imencode('.jpg', online_im)
            frame =  jpeg.tobytes()

            frame_id += 1
        else:
            cam.release()
            break

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n'
        b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n'
        b'\r\n' + frame + b'\r\n')

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
@app.route('/video_feed')
def video_feed():
    return Response(eval_seq(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Flask server shutting down...'


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo_choose.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='jde.1088x608.uncertainty.pt', help='path to weights file')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str,default="webcam0",help='path to the input video')
    parser.add_argument('--img-size', type=str,default=(1088,608),help='video image size')

    opt = parser.parse_args()
    print(opt, end='\n\n')


    # run tracking
    #eval_seq(opt)
    


    w=1088
    h=608

    if(opt.input_video=="webcam0"):
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
        imwidth = opt.img_size[0]
        imheight = opt.img_size[1]
        
        w, h = get_size(vw, vh, imwidth, imheight)
    
    tracker = JDETracker(opt, frame_rate)

    config = json.load(open('config.json', 'r'))
    app.run(host='0.0.0.0', port=config['flask_port'], debug=True)