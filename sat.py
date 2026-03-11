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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.barlow_twins import BarlowTwins
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.rtcl_kmeans import   kmeans
from torch.linalg import norm
from solo.utils.misc import omegaconf_select

import math

class SAT(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
                scale_loss (float): scaling factor of the loss.
        """

        super().__init__(cfg)
        
        #BT
        self.lamb: float = cfg.method_kwargs.lamb
        self.scale_loss: float = cfg.method_kwargs.scale_loss

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        #adv stuff
        self.num_clusters = cfg.adv.num_clusters
        self.warm_up_stage = cfg.adv.warm_up_stage
        # self.warm_up_stage = 1
        self.mixup_alpha = cfg.adv.mixup_alpha
        self.cluster_centers:torch.Tensor= None

        self.pseudo_classifier: nn.Module = nn.Linear(self.features_dim, self.num_clusters)
        
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(BarlowTwins, BarlowTwins).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.0051)
        cfg.method_kwargs.scale_loss = omegaconf_select(cfg, "method_kwargs.scale_loss", 0.024)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params


    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out
    


    def adv_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        return z
    
    def cl_adv_forward(self, X: torch.Tensor) -> Dict[str, Any]:

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        return z

    def pgd_attack(self,inputs, eps=8. / 255., alpha=2. / 255., iters=5, randomInit=True):
        loss = barlow_loss_func
        images = inputs.detach()
        original_images = images.clone().detach()
        # init
        if randomInit:
            delta = torch.rand_like(images) * eps * 2 - eps
        else:
            delta = torch.zeros_like(images)
        delta = torch.nn.Parameter(delta, requires_grad=True)

        for i in range(iters):
 
            z_adv = self.adv_forward(images + delta)
            z_orig = self.forward(original_images)["z"].detach()
            
            self.backbone.zero_grad()
            self.projector.zero_grad()
            cost = loss(z_adv, z_orig.detach(), lamb=self.lamb, scale_loss=self.scale_loss)
            # cost.backward()
            delta_grad = torch.autograd.grad(cost, [delta])[0]

            delta.data = delta.data + alpha * delta_grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

        self.backbone.zero_grad()
        self.projector.zero_grad()

        return (images + delta).detach()
    
    def weight_adv(self):
        w = min(1.0,  self.current_epoch // self.warm_up_stage ) * ((self.current_epoch -self.warm_up_stage ) / (self.max_epochs -self.warm_up_stage ) )
        self.log("weight_adv",w)
        return w
    
    def exp_weight_schedule(self,current_epoch, max_epochs=1000,a = 2):
        """
        Extended exponential scheduling function that increases from 0 to 1 in a convex manner
        over an extended range of x from 0 to x_max.
        
        Args:
        x (float): A value between 0 and x_max inclusive.
        a (float): A positive constant to adjust the steepness of the curve.
        x_max (float): The maximum value of x, defaults to 1000.

        Returns:
        float: The output of the adjusted exponential function.
        """
        if 0 <= current_epoch <= max_epochs:
            scaled_x = current_epoch / max_epochs  # Scale x to be between 0 and 1
            w = (math.exp(a * scaled_x) - 1) / (math.exp(a) - 1)
            self.log("weight_adv",w)
            return w
        else:
            raise ValueError(f"Input x should be between 0 and {max_epochs}.")

    def mixup_target(self, target, lam=1.):
        y1 = target
        y2 = target.flip(0)
        return y1 * lam + y2 * (1. - lam)

    
    def on_after_backward(self):
  
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
        
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = norm(p.grad)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            self.logger.experiment.add_scalar('Gradient_Norm', total_norm, self.trainer.global_step)


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss 
        """
        
        _, X, targets = batch
        #X [augment1, augment2, original_image]
        out = super().training_step(batch, batch_idx)
        z_out = out["z"]
        if isinstance(z_out, (list, tuple)):
            z1 = z_out[0] #projected features of augment1
            z2 = z_out[1] #projected features of augment2
        else:
            raise ValueError("Expected out['z'] to be a list/tuple of views (z1, z2)")

        # barlow twins clean loss
        clean_barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)

        # (same if it were X[2])
        orig_images = X[-1]
        z_orig_images = self.forward(orig_images)["z"].detach()
        
        # generate adversarial image by maximizing Barlow loss against original projection
        adv_x = self.pgd_attack(orig_images)
        adv_z = self.cl_adv_forward(adv_x)
        adv_barlow_loss = barlow_loss_func(adv_z, z_orig_images, lamb=self.lamb, scale_loss=self.scale_loss)

        weight = self.weight_adv()
        return clean_barlow_loss + weight * adv_barlow_loss