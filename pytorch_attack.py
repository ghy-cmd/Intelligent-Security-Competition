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
from cifar10_models.vgg import vgg16_bn
from cifar10_models.inception  import inception_v3
from data import CifarDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from foolbox.distances import l0, l1, l2, linf

from mi_fgsm import LinfMomentumIterativeFastGradientMethod
import sys

import torch
import torch.nn as nn

sys.path.insert(0, '..')
import torchattacks
sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from torchattacks import PGD


def main() -> None:
    images, labels = load_cifar10(n_examples=5)
    print('[Data loaded]')

    device = "cuda"
    model = load_model('Wong2020Fast', norm='Linf').to(device)
    acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('[Model loaded]')
    print('Acc: %2.2f %%'%(acc*100))

    atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    atk.save(data_loader=[(images, labels)], save_path="_transfer.pt")
    adv_loader = atk.load(load_path="_transfer.pt")
    adv_images, labels = iter(adv_loader).next()
    model = load_model('Standard', norm='Linf').to(device)
    print('[Model loaded]')
    acc = clean_accuracy(model, adv_images.to(device), labels.to(device))
    print('Acc: %2.2f %%'%(acc*100))

if __name__ == "__main__":
    main()
    # os.system("/usr/bin/shutdown")
