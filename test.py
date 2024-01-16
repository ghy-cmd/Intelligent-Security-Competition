#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import os

import torchvision.models as models
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

from torchvision import transforms
import torch
from cifar10_models.googlenet import googlenet
from data import CifarDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


def main() -> None:
    batch_size = 100
    num_wokers = 4
    save = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 替换以下行为你的图像文件夹路径
    image_folder = "/root/autodl-tmp/1_L2CarliniWagnerAttack_2"
    label_file = "/root/autodl-tmp/label.txt"

    transform = transforms.Compose([transforms.ToTensor()])
    custom_dataset = CifarDataset(label_file, image_folder, device=device, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # instantiate a model (could also be a TensorFlow or JAX model)
    model = googlenet(pretrained=True, device="cuda:0")
    model.eval()  # for evaluation
    # model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    total_samples = 0
    correct_predictions = 0
    for batch in dataloader:
        images, labels, _ = batch
        predictions = fmodel(images).argmax(axis=-1)
        # 累积样本数和正确预测的样本数
        total_samples += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()
    clean_acc = correct_predictions / total_samples
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")


if __name__ == "__main__":
    main()
    # os.system("/usr/bin/shutdown")
