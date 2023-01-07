
此專案不包含jde pretrained weight、測試影片

## Environment

- Windows
- Python 3.7.15

## Requirements

- ffmpeg
- PyTorch 1.7+


我有進行過的操作
- 針對copy過專案
  - `pip install ffmpeg-python`
  - `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  nvcc --version`
  - `pip install python-ffmpeg-video-streaming`
  - `pip install scipy`
  - `pip install -r requirements.txt`


- 針對jde (可參考jde github python3.7確認可以使用)
  - `pip install cython`
  - `cd cython_bbox-0.1.3`
  - `python setup.py install`
  - `pip install opencv-python`
  - `pip install mptmetics`
  - `pip install numba`
  - `pip install matplotlib`
  - `pip install lap`

- 針對yolox (python 3.8.15 pytorch 1.11可使用)
  下載yolox
  - `git clone https://github.com/Megvii-BaseDetection/YOLOX.git`
  下載 apex-master.zip 放到 yolox folder
  - `https://github.com/NVIDIA/apex`
  - `cd apex-master`
  - `conda install -c nvidia cuda`
  安裝cython_bbox
  - `cd cython_bbox-0.1.3`
  - `python setup.py install`
  
  
  - `pip install -r requirements_stream.txt`
  - `cd yolox`
  - `python setup.py install`
  - `cd ..`


### use yolox pretrained weight 

- github:`https://github.com/Megvii-BaseDetection/YOLOX`
- yolox_s.pth download link:`https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth`

### use JDE pretrained weight JDE-1088x608

- github:`https://github.com/Zhongdao/Towards-Realtime-MOT`
- google download link:`https://drive.google.com/file/d/1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA/view`


## Steps

1. Go to project directory
2. Execute `python http_server`
3. Execute `python server_new.py --input-video 0`(with webcam0) or `python server_new.py --input-video ./hw3_test.mp4`(with local video)
4. Execute `python stream.py`
5. Open `http://<server_ip>:<http_port>/index.html` (`<server_ip>` and `http_port` are defined in `config.json`).

## Reference
- Project Framework: https://github.com/eugene87222/NCTU-Video-Streaming-and-Tracking
- UI design:
  - https://freshman.tech/custom-html5-video/
  - https://www.w3schools.com/css/css_dropdowns.asp
- HLS streaming: https://video.aminyazdanpanah.com/python/start
- Flask server: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00
