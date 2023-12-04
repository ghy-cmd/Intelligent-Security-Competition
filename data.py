import os
from torch.utils.data import Dataset
from PIL import Image
import torch


class CifarDataset(Dataset):
    def __init__(self, label_file, image_folder, device, transform=None):
        self.device = device
        self.image_folder = image_folder
        self.transform = transform

        # 读取标签文件
        self.images = []
        self.labels = []
        with open(label_file, "r") as file:
            for line in file:
                line = line.strip().split()
                image_name, label = line[0], int(line[1])
                self.images.append(image_name)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, label = self.images[index], self.labels[index]

        # 使用PIL库加载图像
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        # 将图像和标签移到设备上
        image = image.to(self.device)
        label = torch.tensor(label).to(self.device)

        # 返回图像和标签
        return image, label, image_name
