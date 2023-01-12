
此專案不包含yolox-s model、測試影片、YOLOX(須替換yolox/utils/visualize.py改為git上傳的visualize.py)
## Environment

- Windows
- Python 3.8.15

## Requirements

- ffmpeg
- PyTorch 1.7+

developer有進行過的操作

- 針對yolox (python 3.8.15 pytorch 1.11可使用)
  - 安裝ffmpeg
  - `pip install ffmpeg-python`
  - `pip install python-ffmpeg-video-streaming`
  - 下載yolox
    - `git clone https://github.com/Megvii-BaseDetection/YOLOX.git`
  - 下載 apex-master.zip
    - `https://github.com/NVIDIA/apex`
    - `cd apex-master`
    - `conda install -c nvidia cuda`
  - 安裝cython_bbox
    - `cd cython_bbox-0.1.3`
    - `python setup.py install`
  - `pip install -r requirements_stream.txt`
  - `cd yolox`
  - `python setup.py install`
  - `cd ..`


### use yolox pretrained weight 

- github:`https://github.com/Megvii-BaseDetection/YOLOX`
- yolox_s.pth download link:`https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth`



## Steps

1. Go to project directory
2. Go to config.json change ip to your sever ip  
3. Execute `python http_server.py`
4. Execute YOLOX Version with webcam0: `python ./server_yolox.py -f YOLOX/exps/default/yolox_s.py -c ./yolox_s.pth --device gpu --input-video webcam0`
5. Execute `python stream.py`
6. Open `http://<server_ip>:<http_port>/index.html` (`<server_ip>` and `http_port` are defined in `config.json`).

## Reference
- Project Framework: https://github.com/eugene87222/NCTU-Video-Streaming-and-Tracking
- UI design:
  - https://freshman.tech/custom-html5-video/
  - https://ithelp.ithome.com.tw/articles/10220978
  - https://www.w3schools.com/css/css_dropdowns.asp
- HLS streaming: 
  - https://video.aminyazdanpanah.com/python/start
  - https://ithelp.ithome.com.tw/articles/10203792
- HLS.js:
  - https://ithelp.ithome.com.tw/articles/10206181
  - https://ithelp.ithome.com.tw/articles/10310697
- Flask server: 
  - https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
  - https://shengyu7697.github.io/python-flask-camera-streaming/

