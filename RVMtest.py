import time
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter
import cv2


def prepare_model():
    device = torch.device('cuda')
    model = torch.jit.load('model/rvm_mobilenetv3_fp32.torchscript')
    model = model.to(device)
    model = torch.jit.freeze(model)
    return model


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()


if __name__ == '__main__':
    reader1 = VideoReader("/home/wave/Videos/dy1.mp4", transform=ToTensor())
    writer1 = VideoWriter('test_result/mobilev3_dance2.flv', frame_rate=30)
    bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).to(torch.float32).cuda()  # 绿背景
    rec = [None] * 4  # 初始循环记忆（Recurrent States）
    downsample_ratio = 0.25  # 下采样比，根据视频调节
    camera = cv2.VideoCapture("/home/wave/Videos/dy1.mp4")

    with torch.no_grad():
        start = time.perf_counter()
        loop_cnt = 0
        while True:
            flag, src = camera.read()
            if not flag:
                break
            src = cv2_frame_to_cuda(src)
        # for src in DataLoader(reader):  # 输入张量，RGB通道，范围为 0～1
            fgr, pha, *rec = model(src, *rec, downsample_ratio)  # 将上一帧的记忆给下一帧
            com = fgr * pha + bgr * (1 - pha)  # 将前景合成到绿色背景
            temp = com.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
            # print(temp.shape)
            # cv2.imshow("show", temp)
            # cv2.waitKey(1)
            # writer.write(com)  # 输出帧
            loop_cnt += 1
    print("{} frames into file".format(loop_cnt))
    end = time.perf_counter()
    print("{} ms ".format((end-start)*1000))

