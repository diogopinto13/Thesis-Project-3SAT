# # Copyright 2023 solo-learn development team.

# # Permission is hereby granted, free of charge, to any person obtaining a copy of
# # this software and associated documentation files (the "Software"), to deal in
# # the Software without restriction, including without limitation the rights to use,
# # copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# # Software, and to permit persons to whom the Software is furnished to do so,
# # subject to the following conditions:

# # The above copyright notice and this permission notice shall be included in all copies
# # or substantial portions of the Software.

# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# # PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# # FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# # OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# # DEALINGS IN THE SOFTWARE.

# import inspect
# import logging
# import os

# import hydra
# import torch
# import copy
# import torch.nn as nn
# from omegaconf import DictConfig, OmegaConf


# from solo.args.linear import parse_cfg
# from solo.data.classification_dataloader import prepare_data
# from solo.methods.base import BaseMethod
# from solo.utils.misc import make_contiguous

# import pytorch_lightning as pl
# import omegaconf
# from solo.utils.misc import  omegaconf_select

# class AdvModel(pl.LightningModule):
#      def __init__(
#         self,
#         backbone: nn.Module,
#         cfg: omegaconf.DictConfig,

#     ):
#         super().__init__()

#         # backbone
#         self.backbone = backbone
#         if hasattr(self.backbone, "inplanes"):
#             features_dim = self.backbone.inplanes
#         else:
#             features_dim = self.backbone.num_features

#         # classifier
#         self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore
#      def forward(self, X: torch.tensor):
#         """Performs forward pass of the frozen backbone and the linear layer for evaluation.

#         Args:
#             X (torch.tensor): a batch of images in the tensor format.

#         Returns:
#             Dict[str, Any]: a dict containing features and logits.
#         """
      
#         feats = self.backbone(X)

#         logits = self.classifier(feats)
#         return logits



# @hydra.main(version_base="1.2")
# def main(cfg: DictConfig):
#     OmegaConf.set_struct(cfg, False)
#     cfg = parse_cfg(cfg)

#     backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

#     # initialize backbone
#     backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
#     if cfg.backbone.name.startswith("resnet"):
#         # remove fc layer
#         backbone.fc = nn.Identity()
#         cifar = cfg.data.dataset in ["cifar10", "cifar100"]
#         if cifar:
#             backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
#             backbone.maxpool = nn.Identity()
    
 
#     model = AdvModel(backbone,cfg=cfg)
#     make_contiguous(model)

#     # load checkpoint
#     ckpt_path = cfg.pretrained_feature_extractor
#     assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    
#     state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#     # load matching keys and ignore unexpected ones from pretraining (projector, pseudo_classifier, ...)
#     model.load_state_dict(state, strict=False)
#     logging.info(f"Loaded {ckpt_path}")

#     if cfg.data.format == "dali":
#         val_data_format = "image_folder"
#     else:
#         val_data_format = cfg.data.format

#     _, val_loader = prepare_data(
#         cfg.data.dataset,
#         train_data_path=cfg.data.train_path,
#         val_data_path=cfg.data.val_path,
#         data_format=val_data_format,
#         batch_size=cfg.optimizer.batch_size,
#         num_workers=cfg.data.num_workers,
#         auto_augment=cfg.auto_augment,
#     )

    
#     l = [batch[0] for  batch in val_loader]
#     x_test = torch.cat(l, 0)
#     l = [batch[1] for batch in val_loader]
#     y_test = torch.cat(l, 0)

    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
#     x_test = x_test.to(device)
#     y_test = y_test.to(device)

#     # Create a plain nn.Module wrapper around the backbone+classifier to avoid
#     # LightningModule properties (like .trainer) being accessed during tracing.
#     class EvalWrapper(nn.Module):
#         def __init__(self, backbone: nn.Module, classifier: nn.Module):
#             super().__init__()
#             self.backbone = backbone
#             self.classifier = classifier

#         def forward(self, x: torch.Tensor):
#             feats = self.backbone(x)
#             return self.classifier(feats)

#     wrapper = EvalWrapper(model.backbone, model.classifier)
#     wrapper = wrapper.to(device)
#     wrapper.eval()

#     # Save pickled wrapper for pipelines that expect a .pth (requires same code to load)
#     try:
#         # save a CPU deepcopy so we don't move the tracing wrapper off-device
#         cpu_wrapper = copy.deepcopy(wrapper).cpu()
#         torch.save(cpu_wrapper, "model_full.pth")
#         logging.info("Saved pickled model as model_full.pth")
#     except Exception:
#         logging.warning("Failed to save pickled model; continuing to TorchScript export")

#     # TorchScript trace (recommended for deployment - produces .pt)
#     example = x_test[:1].to(device)
#     with torch.no_grad():
#         traced = torch.jit.trace(wrapper, example)
#     traced = traced.cpu()
#     torch.jit.save(traced, "3sat_full_adv_model.pt")
#     logging.info("Saved TorchScript model as 3sat_full_adv_model.pt")

# if __name__ == "__main__":
#     main()

# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os

import hydra
import torch
import copy
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


from solo.args.linear import parse_cfg
from solo.data.classification_dataloader import prepare_data
from solo.methods.base import BaseMethod
from solo.utils.misc import make_contiguous

import pytorch_lightning as pl
import omegaconf
from solo.utils.misc import  omegaconf_select

class AdvModel(nn.Module):
     def __init__(
        self,
        backbone: nn.Module,
        cfg: omegaconf.DictConfig,

    ):
        super().__init__()
        # backbone
        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore
     def forward(self, X: torch.tensor):
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
      
        feats = self.backbone(X)

        logits = self.classifier(feats)
        return logits



@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()
    
 
    model = AdvModel(backbone,cfg=cfg)
    make_contiguous(model)

    # load checkpoint
    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # load matching keys and ignore unexpected ones from pretraining (projector, pseudo_classifier, ...)
    model.load_state_dict(state, strict=False)
    logging.info(f"Loaded {ckpt_path}")

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    _, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
    )

    
    l = [batch[0] for  batch in val_loader]
    x_test = torch.cat(l, 0)
    l = [batch[1] for batch in val_loader]
    y_test = torch.cat(l, 0)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Save pickled wrapper for pipelines that expect a .pth (requires same code to load)
    try:
        # save a CPU deepcopy so we don't move the tracing wrapper off-device
        cpu_wrapper = copy.deepcopy(model).cpu()
        torch.save(cpu_wrapper, "model_full.pth")
        logging.info("Saved pickled model as model_full.pth")
    except Exception:
        logging.warning("Failed to save pickled model; continuing to TorchScript export")

    # TorchScript trace (recommended for deployment - produces .pt)
    example = x_test[:1].to(device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced = traced.cpu()
    torch.jit.save(traced, "3sat_full_adv_model.pt")
    logging.info("Saved TorchScript model as 3sat_full_adv_model.pt")

if __name__ == "__main__":
    main()
