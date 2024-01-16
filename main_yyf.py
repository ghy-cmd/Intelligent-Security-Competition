#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import os
import cv2
import foolbox.attacks
import argparse
from tqdm import tqdm
import pytorch_ssim
import foolbox.attacks
import torchvision.models as models
import eagerpy as ep
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.distances import T
from torchvision import transforms
import torch
from cifar10_models.vgg import vgg16_bn,vgg13_bn,vgg19_bn
from cifar10_models.resnet import resnet18,resnet34,resnet50
from cifar10_models.inception  import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.googlenet import googlenet
from cifar10_models.fusion import FusionEnsemble
import time
from cifar10_models.densenet import densenet121,densenet161,densenet169
from data import CifarDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from foolbox.distances import l0, l1, l2, linf
from pytorchcv.model_provider import get_model as ptcv_get_model
from cifar10_models.vit import Vit

from mi_fgsm import LinfMomentumIterativeFastGradientMethod
repo1_model_lists = [
    # densenet121,
    # densenet161,
    # densenet169,
    # googlenet,
    # mobilenet_v2,
    # inception_v3,
    # resnet18,
    # resnet34,
    # resnet50,
    # vgg16_bn,
    # vgg13_bn,
    # vgg19_bn
    ]
pytorchcv_model_lists = [
    # 'densenet250_k24_bc_cifar10',
    # 'diapreresnet164bn_cifar10',
    # 'diaresnet164bn_cifar10',
    # 'preresnet542bn_cifar10',
    # 'nin_cifar10',
    # 'pyramidnet272_a200_bn_cifar10',
    # 'resnet110_cifar10',
    # 'resnet542bn_cifar10',
    # 'resnext272_2x32d_cifar10',
    # 'ror3_164_cifar10',
    # 'sepreresnet542bn_cifar10',
    # 'shakeshakeresnet26_2x32d_cifar10',
    # 'wrn20_10_32bit_cifar10',
    # 'wrn40_8_cifar10',
    # 'xdensenet40_2_k36_bc_cifar10'
]
from ares.utils.registry import registry
robust_train_models = [
    'at_he', # wideres
    'awp', # wrn28-10
    'fast_at', # preact_resnet18
    'featurescatter', #wresnet28_10
    'hydra', # wresnet28_10
    'label_smoothing', # wresnet34_10
    'pre_training', # wresnet34_10
    'robust_overfiting', # wresnet34_10
    'rst', # wresnet34_10
    'trades', # wresnet34_10
]
model_cls = registry.get_model('CifarCLS')
parser = argparse.ArgumentParser()
parser.add_argument("--attack_name", type=str, default=None)
parser.add_argument("--epsilons", type=float, default=5)
args = parser.parse_args()
attack_name = args.attack_name
epsilons = args.epsilons

def main() -> None:
    start_time = time.time()
    batch_size = 25
    num_wokers = 4
    save = True # 保存输出
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 替换以下行为你的图像文件夹路径
    image_folder = "/root/autodl-tmp/images"
    # input_image = Image.open(os.path.join(image_folder,"0.png"))
    # image_folder  = '/root/autodl-tmp/attack_result/0.15_LinfMomentumIterativeFastGradientMethod_1_robustonly'
    label_file = "/root/autodl-tmp/label.txt"

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    custom_dataset = CifarDataset(label_file, image_folder, device=device, transform=transform)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # instantiate a model (could also be a TensorFlow or JAX model)
    # model = vgg16_bn(pretrained=True, device="cuda:0") # TODO:换别的
    # model = inception_v3(pretrained=True, device="cuda:0") # TODO:换别的




    model = FusionEnsemble(model_lists=repo1_model_lists,
                           pytorchcv_model_lists=pytorchcv_model_lists,
                           robust_train_models=robust_train_models,robustbench=[]).eval()
    eval_model = model_cls('fast_at').to(device)
    eval_model.eval()
    # model = Vit().eval()
    # model = models.resnet18(pretrained=True).eval()
    # model = model_cls('featurescatter').to(device)
    model.eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1) ,device=device)

    total_samples = 0
    correct_predictions = 0
    for batch in dataloader: # 原始正确率
        images, labels, _ = batch
        predictions = fmodel(images)
        predictions = predictions.argmax(axis=-1)
        # 累积样本数和正确预测的样本数
        total_samples += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()
    clean_acc = correct_predictions / total_samples
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    # attack = foolbox.attacks.LinfFastGradientAttack(random_start=False)
    # attack = foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack(repeats=100, check_trivial=True)
    # attack = foolbox.attacks.SaltAndPepperNoiseAttack(steps=100, across_channels=True, channel_axis=None)
    # attack = foolbox.attacks.LinfDeepFoolAttack(steps=50, candidates=10, overshoot=0.02, loss="logits")
    if attack_name == "L2ContrastReductionAttack":
            attack = foolbox.attacks.L2ContrastReductionAttack()
    elif attack_name == "VirtualAdversarialAttack":
        attack = foolbox.attacks.VirtualAdversarialAttack(steps=1000)
    elif attack_name == "DDNAttack":
        attack = foolbox.attacks.DDNAttack()
    elif attack_name == "L2ProjectedGradientDescentAttack":
        attack = foolbox.attacks.L2ProjectedGradientDescentAttack()
    elif attack_name == "LinfProjectedGradientDescentAttack":
        attack = foolbox.attacks.LinfProjectedGradientDescentAttack()
    elif attack_name == "LinfBasicIterativeAttack":
        attack = foolbox.attacks.LinfBasicIterativeAttack()
    elif attack_name == "L2FastGradientAttack":
        attack = foolbox.attacks.L2FastGradientAttack()
    elif attack_name == "LinfFastGradientAttack":
        attack = foolbox.attacks.LinfFastGradientAttack()
    elif attack_name == "L2AdditiveGaussianNoiseAttack":
        attack = foolbox.attacks.L2AdditiveGaussianNoiseAttack()
    elif attack_name == "L2AdditiveUniformNoiseAttack":
        attack = foolbox.attacks.L2AdditiveUniformNoiseAttack()
    elif attack_name == "L2ClippingAwareAdditiveGaussianNoiseAttack":
        attack = foolbox.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack()
    elif attack_name == "L2ClippingAwareAdditiveUniformNoiseAttack":
        attack = foolbox.attacks.L2ClippingAwareAdditiveUniformNoiseAttack()
    elif attack_name == "LinfAdditiveUniformNoiseAttack":
        attack = foolbox.attacks.LinfAdditiveUniformNoiseAttack()
    elif attack_name == "L2RepeatedAdditiveGaussianNoiseAttack":
        attack = foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack()
    elif attack_name == "LinfMomentumIterativeFastGradientMethod":
        attack = LinfMomentumIterativeFastGradientMethod()
    elif attack_name == "L2RepeatedAdditiveUniformNoiseAttack":
        attack = foolbox.attacks.L2RepeatedAdditiveUniformNoiseAttack()
    elif attack_name == "L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack":
        attack = foolbox.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack()
    elif attack_name == "L2ClippingAwareRepeatedAdditiveUniformNoiseAttack":
        attack = foolbox.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack()
    elif attack_name == "LinfRepeatedAdditiveUniformNoiseAttack":
        attack = foolbox.attacks.LinfRepeatedAdditiveUniformNoiseAttack()
    elif attack_name == "InversionAttack":
        attack = foolbox.attacks.InversionAttack(distance=linf)
    elif attack_name == "BinarySearchContrastReductionAttack":
        attack = foolbox.attacks.BinarySearchContrastReductionAttack(distance=linf)
    elif attack_name == "LinearSearchContrastReductionAttack":
        attack = foolbox.attacks.LinearSearchContrastReductionAttack(distance=linf)
    elif attack_name == "L2CarliniWagnerAttack":
        attack = foolbox.attacks.L2CarliniWagnerAttack()
    elif attack_name == "NewtonFoolAttack":
        attack = foolbox.attacks.NewtonFoolAttack()
    elif attack_name == "EADAttack":
        attack = foolbox.attacks.EADAttack()
    elif attack_name == "GaussianBlurAttack":
        attack = foolbox.attacks.GaussianBlurAttack(distance=linf)
    elif attack_name == "L2DeepFoolAttack":
        attack = foolbox.attacks.L2DeepFoolAttack()
    elif attack_name == "LinfDeepFoolAttack":
        attack = foolbox.attacks.LinfDeepFoolAttack()
    elif attack_name == "SaltAndPepperNoiseAttack":
        attack = foolbox.attacks.SaltAndPepperNoiseAttack()
    elif attack_name == "LinearSearchBlendedUniformNoiseAttack":
        attack = foolbox.attacks.LinearSearchBlendedUniformNoiseAttack(distance=linf)
    elif attack_name == "BinarizationRefinementAttack":
        attack = foolbox.attacks.BinarizationRefinementAttack(distance=linf)
    elif attack_name == "DatasetAttack":
        attack = foolbox.attacks.DatasetAttack(distance=linf)
    elif attack_name == "BoundaryAttack":
        attack = foolbox.attacks.BoundaryAttack()
    elif attack_name == "L0BrendelBethgeAttack":
        attack = foolbox.attacks.L0BrendelBethgeAttack()
    elif attack_name == "L1BrendelBethgeAttack":
        attack = foolbox.attacks.L1BrendelBethgeAttack()
    elif attack_name == "L2BrendelBethgeAttack":
        attack = foolbox.attacks.L2BrendelBethgeAttack()
    elif attack_name == "LinfinityBrendelBethgeAttack":
        attack = foolbox.attacks.LinfinityBrendelBethgeAttack()
    elif attack_name == "L0FMNAttack":
        attack = foolbox.attacks.L0FMNAttack()
    elif attack_name == "L1FMNAttack":
        attack = foolbox.attacks.L1FMNAttack()
    elif attack_name == "L2FMNAttack":
        attack = foolbox.attacks.L2FMNAttack()
    elif attack_name == "LInfFMNAttack":
        attack = foolbox.attacks.LInfFMNAttack()
    elif attack_name == "PointwiseAttack":
        attack = foolbox.attacks.PointwiseAttack()
    elif attack_name == "FGM":
        attack = foolbox.attacks.FGM()
    elif attack_name == "FGSM":
        attack = foolbox.attacks.FGSM()
    elif attack_name == "L2PGD":
        attack = foolbox.attacks.L2PGD()
    elif attack_name == "LinfPGD":
        attack = foolbox.attacks.LinfPGD()
    elif attack_name == "PGD":
        attack = foolbox.attacks.PGD()
    base_directory = "result/" + str(epsilons)
    # 如果目录已经存在，创建一个新的目录
    folder_suffix = 1
    save_directory = base_directory + '_' + attack_name + '_' + str(folder_suffix)+'_robustonly_all_in'
    while os.path.exists(save_directory):
        folder_suffix += 1
        save_directory = base_directory + '_' + attack_name + '_' + str(folder_suffix)+'_robustonly_all_in'
    # 创建新目录
    os.makedirs(save_directory)
    batch_num = 0
    robust_accuracy = torch.zeros(1).to(device)
    for batch in tqdm(dataloader):
        batch_num += 1
        images, labels, image_names = batch
        #labels.requires_grad = True
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
                # adv_image = transforms.Resize(input_image.size)(adv_image)
                save_path = os.path.join(save_directory, image_names[i])
                adv_image.save(save_path)
    robust_accuracy = robust_accuracy / batch_num
    print("robust accuracy for perturbations with")
    # for eps, acc in zip(epsilons, robust_accuracy):
    #     print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
    print(f"  Linf norm ≤ {epsilons:<6}: {robust_accuracy.item() * 100:4.1f} %")
    image_folder = save_directory
    label_file = "/root/autodl-tmp/label.txt"
    adv_dataset = CifarDataset(label_file, image_folder, device=device, transform=transform)
    dataloader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)
    total_samples = 0
    correct_predictions = 0
    for batch in dataloader: # 原始正确率
        images, labels, _ = batch
        predictions = eval_model(images).argmax(axis=-1)
        # 累积样本数和正确预测的样本数
        total_samples += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()
    clean_acc = correct_predictions / total_samples
    total_ssim = 0
    for i in range(500):
        npImg1 = cv2.imread(os.path.join(save_directory, str(i) + ".png"))
        img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
        img1 = img1.to(device)
        npImg2 = cv2.imread(os.path.join("/root/autodl-tmp/images", str(i) + ".png"))
        img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0
        img2 = img2.to(device)
        ssim_value = pytorch_ssim.ssim(img1, img2).item()
        total_ssim += ssim_value
    ssim = total_ssim / 500
    print(ssim)
    score = 100*(1-clean_acc)*ssim
    print(f"score:  {score:.1f}")
    end_time = time.time()
    with open(f"result/result.txt", 'a') as file:
        file.write(f"{attack_name}:  epsilon: {epsilons}  score:  {score:.1f} acc:  {clean_acc:.6f} ssim:  {ssim:.6f}   cost_time:{end_time-start_time}\n")


if __name__ == "__main__":
    main()
    # os.system("/usr/bin/shutdown")
