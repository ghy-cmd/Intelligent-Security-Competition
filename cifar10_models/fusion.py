import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from ares.utils.registry import registry
from robustbench.utils import load_model

model_cls = registry.get_model('CifarCLS')
class FusionEnsemble(nn.Module):
    def __init__(self, 
                 model_lists,
                 pytorchcv_model_lists,
                 robust_train_models, 
                 robustbench,
                 device='cuda'):
        super(FusionEnsemble, self).__init__()
        self.seqs = nn.ModuleList([model(pretrained=True).to(device) for model in model_lists])
        for model_name in pytorchcv_model_lists:
            self.seqs.append(ptcv_get_model(model_name, pretrained=True).to(device))
        for model_name in robust_train_models:
            self.seqs.append(model_cls(model_name).to(device))
        for model_name in robustbench:
            if model_name in [
                'Augustin2020Adversarial_34_10','Wang2023Better_WRN-28-10'
            ]:
                threat_model = 'L2'
            else:
                threat_model = 'Linf'
            model = load_model(model_name=model_name, dataset='cifar10', threat_model=threat_model)
            self.seqs.append(model.to(device))        

        self.model_num = len(self.seqs)
        print(self.model_num)
    def forward(self, x):
        xx = None
        for model in self.seqs:
            if xx == None:
                xx = model(x)
            else:
                xx += model(x)
        
        return xx / self.model_num