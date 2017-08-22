import cv2
import time
import os
import sys
import torch as th
from PIL import Image
from torchvision import transforms
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# video
print(opt.input_video)
video_capture = cv2.VideoCapture(opt.input_video)
W = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
W, H = 640, 480
#W, H = 128, 128
#W, H = 256, 256
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(opt.name+'.avi', fourcc, 20.0, (W, H))


model = create_model(opt)
BUFFER = 14
# test
it = 0
while True:
    it += 1
    t = time.time()
    x = []
    for b in range(BUFFER):
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
    
    
        T = transforms.Compose([
            transforms.Scale([W, H]), 
            transforms.ToTensor(),
            #lambda x: x * 2. - 1.
        ])
        x += [T(img)[None]]
    if len(x) == 0: break
    x = th.cat(x, 0)
    if opt.gpu_ids[0] > -1:
        x = x.cuda(opt.gpu_ids[0])
    y = -model.forward_external(x, 'BtoA')
    for frame in y:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_video.write(frame)
    
    print('processed frame... %4d   FPS: %5.2f,' % (
        it*BUFFER, BUFFER/(time.time()-t)))
    
    if not ret:
        break

    
out_video.release()
video_capture.release()
print("Ended!")
