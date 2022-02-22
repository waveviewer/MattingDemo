import time
from inference_utils import VideoReader
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import av


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()


def av_read_rtmp(video_src, frames_count):
    container = av.open(video_src, 'r')
    idx = 0
    start = time.perf_counter()
    for frame in container.decode(video=0):
        idx += 1
        if idx == 10:
           start = time.perf_counter()
        elif idx > frames_count+10:
            break
        img = frame.to_ndarray(format='rgb24')
        src = ToTensor()(Image.fromarray(img)).unsqueeze_(0).cuda()
        pass
    end = time.perf_counter()
    duration = (end - start) * 1000
    print("av package reads {} loops, time is {} ms".format(frames_count, duration))
    pass

def av_read_frame(video_src, frames_count):
    reader = VideoReader(video_src, transform=ToTensor())
    start = time.perf_counter()
    for i in range(frames_count):
        src = reader.__getitem__(i)
    end = time.perf_counter()
    duration = (end - start) * 1000
    print("av package reads {} loops, time is {} ms".format(frames_count, duration))
    pass


def cv2_read_frame(video_src, frames_count):
    reader = cv2.VideoCapture(video_src)
    start = time.perf_counter()
    idx = 0
    while idx < frames_count:
        flag, src = reader.read()
        if not flag:
            continue
        idx += 1
        if idx == 10:
            start = time.perf_counter()
        if idx > frames_count+10:
            break
        src = cv2_frame_to_cuda(src)
    end = time.perf_counter()
    duration = (end - start) * 1000
    print("opencv read {} loops, time is {} ms".format(frames_count, duration))
    pass


if __name__ == '__main__':
    loopcount = 100
    rtmp_addr = "rtmp://127.0.0.1:1935/live/1"
    av_read_rtmp(rtmp_addr, 100)
    # av_read_frame(rtmp_addr, loopcount)
    cv2_read_frame(rtmp_addr, loopcount)
    pass
