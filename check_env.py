import torch

if __name__ == '__main__':
    print(torch.__version__)
    print("CUDA in use {}".format(torch.cuda.is_available()))
    print("GPU count is {}".format(torch.cuda.device_count()))
    print("Device name is {}".format(torch.cuda.get_device_name(0)))
