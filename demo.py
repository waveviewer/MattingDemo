import time
from multiprocessing import Process
import numpy as np
import torch
from torchvision.transforms import ToTensor
import cv2
import os
from PIL import Image

device = torch.device('cuda')
precision = torch.float32


def prepare_model(model_path):
    loaded_model = torch.jit.load(model_path)
    loaded_model.backbone_scale = 0.25
    loaded_model.refine_mode = 'sampling'
    loaded_model.refine_sample_pixels = 80_000
    loaded_model = loaded_model.to(device)
    return loaded_model


def prepare_web_camera():
    camera_capture = cv2.VideoCapture(0)
    ret = camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    print("Frame width set to 640 : ", ret)
    ret = camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Frame height set to 480 : ", ret)
    ret = camera_capture.set(cv2.CAP_PROP_FPS, 30)
    print("FPS set to 30 : ", ret)
    print("Camera is opened : ", camera_capture.isOpened())
    # for i in range(15):
    #     ret, temp = camera_capture.read()
    # cv2.imwrite("Background/back.png", temp)
    # print("Write background to file")
    return camera_capture


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0).cuda()


def gpu_usage_test():
    model = prepare_model(r"D:\BackgroundMattingV2\TorchScript\torchscript_resnet50_fp32.pth")
    background = cv2.imread("Background/back.png")
    bgr = cv2_frame_to_cuda(background)

    display = np.full((960, 1920, 3), np.array([120, 255, 155]), dtype=np.uint8)
    display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("show"+str(os.getpid()), cv2.WINDOW_NORMAL)
    with torch.no_grad():
        start = time.process_time()
        for i in range(1000):
            src = torch.rand(1, 3, 480, 640).to(precision).to(device)
            pha, fgr = model(src, bgr)[:2]
            res = pha * fgr + (1 - pha) * torch.tensor([120 / 255, 255 / 255, 155 / 255], device='cuda').view(1, 3, 1,
                                                                                                              1)
            res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]

            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            display[0:480, 0:640] = res
            display[480:960, 640:1280] = res
            display[0:480, 1280:1920] = res
            cv2.imshow("show"+str(os.getpid()), display)
            cv2.waitKey(1)
    all_end = time.process_time()
    print("finish ",os.getpid())
    print("1000 loops time is {} ms".format((all_end - start) * 1000))
    pass


if __name__ == '__main__':
    p1 = Process(target=gpu_usage_test)
    p1.start()
    print("P1 start")
    p2 = Process(target=gpu_usage_test)
    p2.start()
    print("P2 start")
    # p3 = Process(target=gpu_usage_test)
    # p3.start()
    # print("P3 start")
    p1.join()
    p2.join()
    # p3.join()
    print("All finish")

    # camera = prepare_web_camera()
    #
    # model = prepare_model(r"D:\BackgroundMattingV2\TorchScript\torchscript_resnet50_fp32.pth")
    # background = cv2.imread("Background/back.png")
    # bgr = cv2_frame_to_cuda(background)
    # # bgr = bgr.half()
    # display = np.full((960, 1920, 3), np.array([120, 255, 155]), dtype=np.uint8)
    # display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    # # cv2.namedWindow("show", cv2.WINDOW_NORMAL)
    #
    # with torch.no_grad():
    #     while True:
    #         read_frame_start = time.perf_counter()
    #         flat, temp = camera.read()
    #         read_frame_end = time.perf_counter()
    #         print("Read a frame from camera : {} ms".format((read_frame_end - read_frame_start)*1000))
    #
    #         mat2tensor_start = time.perf_counter()
    #         src = cv2_frame_to_cuda(temp)
    #         # src = src.half()
    #         mat2tensor_end = time.perf_counter()
    #         print("mat2tensor : {} ms".format((mat2tensor_end - mat2tensor_start) * 1000))
    #
    #         model_start = time.perf_counter()
    #         pha, fgr = model(src, bgr)[:2]
    #         torch.cuda.synchronize()
    #         model_end = time.perf_counter()
    #         print("model process to cpu: {} ms".format((model_end - model_start) * 1000))
    #         res = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
    #         res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
    #
    #         result_start = time.perf_counter()
    #         res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    #         display[0:480, 0:640] = res
    #         display[480:960, 640:1280] = res
    #         display[0:480, 1280:1920] = res
    #         result_end = time.perf_counter()
    #         print("result process: {} ms".format((result_end - result_start) * 1000))
    #         print("Full time of a loop is {} ms".format((result_end - read_frame_start)*1000))
    #
    #         # show_start = time.perf_counter()
    #         # cv2.imshow("show", display)
    #         # key = cv2.waitKey(1)
    #         # if key == 27:
    #         #     break
    #         # show_end = time.perf_counter()
    #         # print("rendering time is {} ms".format((show_end - show_start) * 1000))
    #         pass
    #
    # camera.release()
    # print("All finished")
    # pass