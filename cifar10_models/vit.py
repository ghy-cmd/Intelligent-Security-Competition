import torch.nn as nn
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import transforms
from PIL import Image
import numpy as np
class Vit(nn.Module):
    def __init__(self,path='cifar10_models/state_dicts/model_hub/vit-base-patch16-224-cifar10', device='cuda'):
        super(Vit, self).__init__()
        #self.feature_extractor = ViTFeatureExtractor.from_pretrained(path,do_rescale=False)
        self.model = ViTForImageClassification.from_pretrained(path).to(device)
        self.device = device
        for i,(name,param) in enumerate(self.model.named_parameters()):
            assert param.requires_grad == True

    def forward(self, x):
        #inputs = self.feature_extractor(images=x, return_tensors="pt").to(self.device)
        # x = x.detach().cpu().numpy()
        # x = (x * 255).astype(np.uint8)
        # inputs = []
        # # 循环处理每个图像
        # for i in range(x.shape[0]):
        #     # 从张量中获取一张图像的数据
        #     adv_image_data = x[i]
        #     # 创建Image对象
        #     adv_image = Image.fromarray(adv_image_data.transpose(1, 2, 0))  # 调整通道顺序
        #     adv_image = transforms.Resize((224, 224))(adv_image)
        #     save_path = 'test.png'
        #     adv_image.save(save_path)
        #     inputs.append(np.array(adv_image))
        # inputs = np.array(inputs)
        # inputs = inputs.transpose(0,3,1,2)
        # inputs = torch.from_numpy(inputs).to(self.device) / 255.0
        inputs = x
        outputs = self.model(inputs)
        return outputs.logits