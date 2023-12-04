#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import os

import foolbox.attacks
import torchvision.models as models
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.distances import T

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
    save = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 替换以下行为你的图像文件夹路径
    image_folder = "/root/autodl-tmp/images"
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

    # apply the attack
    # attack = foolbox.attacks.LinfFastGradientAttack(random_start=False)
    attack = foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack(repeats=100, check_trivial=True)
    # attack = foolbox.attacks.SaltAndPepperNoiseAttack(steps=100, across_channels=True, channel_axis=None)
    # attack = foolbox.attacks.LinfDeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss="logits")

    epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    epsilons = 10
    base_directory = "/root/autodl-tmp/" + str(epsilons)
    # 如果目录已经存在，创建一个新的目录
    folder_suffix = 1
    save_directory = base_directory + '_' + str(folder_suffix)
    while os.path.exists(save_directory):
        folder_suffix += 1
        save_directory = base_directory + '_' + str(folder_suffix)
    # 创建新目录
    os.makedirs(save_directory)
    batch_num = 0
    robust_accuracy = torch.zeros(1).to(device)
    for batch in dataloader:
        batch_num += 1
        images, labels, image_names = batch

        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        robust_accuracy = robust_accuracy + 1 - success.float().mean(axis=-1)
        # temp = robust_accuracy / batch_num
        # print(f"  Linf norm ≤ {epsilons:<6}: {temp.item() * 100:4.1f} %")
        if save:
            clipped_advs = clipped_advs.cpu().numpy()
            clipped_advs = (clipped_advs * 255).astype(np.uint8)
            # 循环处理每个图像
            for i in range(clipped_advs.shape[0]):
                # 从张量中获取一张图像的数据
                adv_image_data = clipped_advs[i]
                # 创建Image对象
                adv_image = Image.fromarray(adv_image_data.transpose(1, 2, 0))  # 调整通道顺序
                save_path = os.path.join(save_directory, image_names[i])
                adv_image.save(save_path)
    robust_accuracy = robust_accuracy / batch_num
    print("robust accuracy for perturbations with")
    # for eps, acc in zip(epsilons, robust_accuracy):
    #     print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    print(f"  Linf norm ≤ {epsilons:<6}: {robust_accuracy.item() * 100:4.1f} %")


if __name__ == "__main__":
    main()
    # os.system("/usr/bin/shutdown")
