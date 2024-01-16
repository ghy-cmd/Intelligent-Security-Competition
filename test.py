#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import os
import cv2
import pytorch_ssim
from cifar10_models.resnet import resnet50
import torchvision.models as models
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from cifar10_models.vgg import vgg16_bn
from cifar10_models.inception  import inception_v3
from torchvision import transforms
import torch
from cifar10_models.googlenet import googlenet
from data import CifarDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from cifar10_models.vit import Vit
from ares.utils.registry import registry
from robustbench.utils import load_model

model_cls = registry.get_model('CifarCLS')
def main() -> None:
    batch_size = 8
    num_wokers = 4
    save = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 替换以下行为你的图像文件夹路径
    image_folder = "/root/autodl-tmp/attack_result/2_autoattack_final33_1_robustonly"
    label_file = "/root/autodl-tmp/label.txt"

    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    custom_dataset = CifarDataset(label_file, image_folder, device=device, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # instantiate a model (could also be a TensorFlow or JAX model)
    # model = vgg16_bn(pretrained=True, device="cuda:0")
    model = ptcv_get_model('pyramidnet272_a200_bn_cifar10', pretrained=True).to(device)
    # model = resnet50(pretrained=True,device="cuda:0")
    # model = model_cls('fast_at').to(device)
    # model = load_model(model_name='Cui2023Decoupled_WRN-34-10', dataset='cifar10', threat_model='Linf')

    model.eval()  # for evaluation
    # model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), device=device)
    total_samples = 0
    correct_predictions = 0
    for batch in dataloader:
        images, labels, _ = batch
        predictions = fmodel(images)
        predictions = predictions.argmax(axis=-1)
        # 累积样本数和正确预测的样本数
        total_samples += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()
    clean_acc = correct_predictions / total_samples
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    ssim_order = {}
    total_ssim = 0
    for i in range(500):
        npImg1 = cv2.imread(os.path.join(image_folder, str(i) + ".png"))
        # npImg1 = cv2.resize(npImg1,(32,32))
        img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        img1 = img1.to(device)
        npImg2 = cv2.imread(os.path.join("/root/autodl-tmp/images", str(i) + ".png"))
        img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0
        img2 = img2.to(device)
        ssim_value = pytorch_ssim.ssim(img1, img2).item()
        if ssim_value < 0.93:
            ssim_order[i] = ssim_value
        total_ssim += ssim_value
    print(sorted(ssim_order.items(), key = lambda kv:(kv[1], kv[0])))   
    print(ssim_order.keys())
    print(len(ssim_order.keys()))
    ssim = total_ssim / 500
    score = 100*(1-clean_acc)*ssim
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    print(f"ssim:  {ssim * 100:.1f} %")
    print(f"score:  {score:.1f}")

if __name__ == "__main__":
    main()
    # os.system("/usr/bin/shutdown")
