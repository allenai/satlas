import torch
import torchvision
from vit_pytorch import ViT

class Model(torch.nn.Module):
    def __init__(self, info):
        super(Model, self).__init__()

        self.task_specs = info['tasks']
        self.tasks = [spec['Task'] for spec in self.task_specs]

        self.model_config = info['config']
        self.num_channels = self.model_config.get('NumChannels', len(info['channels']))

        self.num_outputs = 0
        self.task_output_offsets = [0]
        for task in self.tasks:
            cur_outputs = len(task['categories'])
            self.num_outputs += cur_outputs
            self.task_output_offsets.append(self.num_outputs)

        backbone_name = self.model_config['Backbone']
        pretrained = self.model_config.get('Pretrained', True)

        if backbone_name.startswith('resnet'):
            if backbone_name == 'resnet18':
                if pretrained:
                    self.backbone = torchvision.models.resnet.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
                else:
                    self.backbone = torchvision.models.resnet.resnet18()
            elif backbone_name == 'resnet34':
                if pretrained:
                    self.backbone = torchvision.models.resnet.resnet34(weights=torchvision.models.resnet.ResNet34_Weights.DEFAULT)
                else:
                    self.backbone = torchvision.models.resnet.resnet34()
            elif backbone_name == 'resnet50':
                if pretrained:
                    self.backbone = torchvision.models.resnet.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
                else:
                    self.backbone = torchvision.models.resnet.resnet50()
            elif backbone_name == 'resnet101':
                if pretrained:
                    self.backbone = torchvision.models.resnet.resnet101(weights=torchvision.models.resnet.ResNet101_Weights.DEFAULT)
                else:
                    self.backbone = torchvision.models.resnet.resnet101()
            elif backbone_name == 'resnet152':
                if pretrained:
                    self.backbone = torchvision.models.resnet.resnet152(weights=torchvision.models.resnet.ResNet152_Weights.DEFAULT)
                else:
                    self.backbone = torchvision.models.resnet.resnet152()

            self.backbone.conv1 = torch.nn.Conv2d(self.num_channels, self.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, self.num_outputs)

        elif backbone_name.startswith('swin'):
            if backbone_name == 'swin_t':
                if pretrained:
                    self.backbone = torchvision.models.swin_t(weights=torchvision.models.swin_transformer.Swin_T_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_t()
            elif backbone_name == 'swin_s':
                if pretrained:
                    self.backbone = torchvision.models.swin_s(weights=torchvision.models.swin_transformer.Swin_S_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_s()
            elif backbone_name == 'swin_b':
                if pretrained:
                    self.backbone = torchvision.models.swin_b(weights=torchvision.models.swin_transformer.Swin_B_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_b()
            elif backbone_name == 'swin_v2_t':
                if pretrained:
                    self.backbone = torchvision.models.swin_v2_t(weights=torchvision.models.swin_transformer.Swin_V2_T_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_v2_t()
            elif backbone_name == 'swin_v2_s':
                if pretrained:
                    self.backbone = torchvision.models.swin_v2_s(weights=torchvision.models.swin_transformer.Swin_V2_S_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_v2_s()
            elif backbone_name == 'swin_v2_b':
                if pretrained:
                    self.backbone = torchvision.models.swin_v2_b(weights=torchvision.models.swin_transformer.Swin_V2_B_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = torchvision.models.swin_v2_b()

            self.backbone.features[0][0] = torch.nn.Conv2d(self.num_channels, self.backbone.features[0][0].out_channels, kernel_size=(4, 4), stride=(4, 4))
            self.backbone.head = torch.nn.Linear(self.backbone.head.in_features, self.num_outputs)

        elif backbone_name == 'vit':
            self.backbone = ViT(image_size=512,patch_size=32,num_classes=self.num_outputs,dim=1024,depth=6,heads=16,mlp_dim=2048,dropout=0.1,emb_dropout=0.1)

    def forward(self, image_list, targets=None, selected_task=None):
        images = torch.stack(image_list, dim=0)
        logits = self.backbone(images)

        outputs = []
        losses = []

        for task_idx, task in enumerate(self.tasks):
            cur_logits = logits[:, self.task_output_offsets[task_idx]:self.task_output_offsets[task_idx+1]]

            if task['type'] == 'classification':
                cur_outputs = torch.nn.functional.softmax(cur_logits, dim=1)

                if targets is not None:
                    cur_valid = torch.stack([target[task_idx]['valid'] for target in targets], dim=0)
                    cur_targets = torch.cat([target[task_idx]['label'] for target in targets], dim=0).to(torch.long)
                    loss = torch.nn.functional.cross_entropy(cur_logits, cur_targets, reduction='none') * cur_valid
                    losses.append(loss.mean())
            elif task['type'] == 'multi-label-classification':
                cur_outputs = torch.sigmoid(cur_logits)

                if targets is not None:
                    cur_valid = torch.stack([target[task_idx]['valid'] for target in targets], dim=0)
                    cur_targets = torch.stack([target[task_idx]['labels'] for target in targets], dim=0).to(torch.float32)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(cur_logits, cur_targets.squeeze(), reduction='none') * cur_valid
                    losses.append(loss.mean())

            outputs.append(cur_outputs)

        if len(losses) > 0:
            losses = torch.stack(losses, dim=0)
        else:
            losses = torch.tensor(0, device=images.device, dtype=torch.float32)

        return outputs, losses
