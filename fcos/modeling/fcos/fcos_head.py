import math
import torch.nn as nn
import torch

from detectron2.layers import Conv2d, DeformConv, ShapeSpec
from fcos.layers import Scale, normal_init
from typing import List


class FCOSHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    In our Implementation, schemetic structure is as following:

                                    /-> logits
                    /-> cls convs ->
                   /                \-> centerness
    shared convs ->
                    \-> reg convs -> regressions
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_channels = input_shape[0].channels  # Fanchen: used
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES  # Fanchen: used
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_shared_convs = cfg.MODEL.FCOS.NUM_SHARED_CONVS  # Fanchen: used
        self.num_stacked_convs = cfg.MODEL.FCOS.NUM_STACKED_CONVS  # Fanchen: used
        self.prior_prob = cfg.MODEL.FCOS.PRIOR_PROB  # Fanchen: used
        self.use_deformable = cfg.MODEL.FCOS.USE_DEFORMABLE  # Fanchen: used
        self.norm_layer = cfg.MODEL.FCOS.NORM  # Fanchen: used
        self.ctr_on_reg = cfg.MODEL.FCOS.CTR_ON_REG  # Fanchen: used
        self.norm_reg_tgt = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS  # Fanchen: new added
        # fmt: on

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initializes six convolutional layers for FCOS head and a scaling layer for bbox predictions.
        """
        activation = nn.ReLU()
        """ your code starts here """
        norm = nn.GroupNorm(32, self.in_channels) if self.norm_layer == 'GN' else None

        self.shared_convs = nn.Sequential(
            *[Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
              for _ in range(self.num_shared_convs)])

        # Fanchen: cls_convs: [H*W*256 --> H*W*256] * 4 + [H*W*256 --> H*W*C(cls) / H*W*1(ctns)]
        self.cls_convs = nn.Sequential(
            *[Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
              for _ in range(self.num_stacked_convs)]) if not self.use_deformable else nn.Sequential(
            *[Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
              for _ in range(self.num_stacked_convs - 1)] + [
                 DeformConv(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm,
                            activation=activation)])
        # Fanchen: Following the original implement, the last layer of stacked convs is DeformConv (if applied)

        # Fanchen: reg_convs: [H*W*256 --> H*W*256] * 4 + [H*W*256 --> H*W*4]
        self.reg_convs = nn.Sequential(
            *[Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
              for _ in range(self.num_stacked_convs)]) if not self.use_deformable else nn.Sequential(
            *[Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm, activation=activation)
              for _ in range(self.num_stacked_convs - 1)] + [
                 DeformConv(self.in_channels, self.in_channels, kernel_size=3, padding=1, norm=norm,
                            activation=activation)])

        self.cls_logits = Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.bbox_pred = Conv2d(self.in_channels, 4, kernel_size=3, padding=1)
        self.centerness = Conv2d(self.in_channels, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleList([Scale() for _ in range(5)])
        """ your code ends here """

    def _init_weights(self):
        for modules in [
            self.shared_convs, self.cls_convs, self.reg_convs,
            self.cls_logits, self.bbox_pred, self.centerness
        ]:
            # weight initialization with mean=0, std=0.01
            for module in modules.modules():
                if isinstance(module, (Conv2d, DeformConv)):
                    normal_init(module, mean=0, std=0.01)

        # initialize the bias for classification logits
        bias_cls = -math.log((1.0 - self.prior_prob) / self.prior_prob)
        # Fanchen: from the paper of RetinaNet, b = -log((1-pi)/pi)
        # calculate proper value that makes cls_probability with `self.prior_prob`
        # In other words, make the initial 'sigmoid' activation of cls_logits as `self.prior_prob`
        # by controlling bias initialization
        nn.init.constant_(self.cls_logits.bias, bias_cls)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            cls_scores (list[Tensor]): list of #feature levels, each has shape (N, C, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of C object classes.
            bbox_preds (list[Tensor]): list of #feature levels, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (l, t, r, b) box regression values for
                every position of featur map. These values are the distances from
                a specific point to each (left, top, right, bottom) edge
                of the corresponding ground truth box that the point belongs to.
            centernesses (list[Tensor]): list of #feature levels, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness logits, where these values used to
                downweight the bounding box scores far from the center of an object.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []

        # Fanchen: self.in_features = _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
        # Fanchen: features = [features[f] for f in self.in_features]
        # Fanchen: self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]

        for feat_level, feature in enumerate(features):
            """ your code starts here """
            feature = self.shared_convs(feature)

            cls = self.cls_convs(feature)
            reg = self.reg_convs(feature)

            cls_scores.append(self.cls_logits(cls))
            centernesses.append(self.centerness(cls) if not self.ctr_on_reg else self.centerness(reg))
            bbox = self.scales[feat_level](self.bbox_pred(reg))
            bbox_preds.append(torch.exp(bbox))
            """ your code ends here """
        return cls_scores, bbox_preds, centernesses
