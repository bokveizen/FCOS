import math
import torch.nn as nn

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
        self.in_channels = input_shape[0].channels
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_shared_convs = cfg.MODEL.FCOS.NUM_SHARED_CONVS
        self.num_stacked_convs = cfg.MODEL.FCOS.NUM_STACKED_CONVS
        self.prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.use_deformable = cfg.MODEL.FCOS.USE_DEFORMABLE
        self.norm_layer = cfg.MODEL.FCOS.NORM
        self.ctr_on_reg = cfg.MODEL.FCOS.CTR_ON_REG
        # fmt: on

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initializes six convolutional layers for FCOS head and a scaling layer for bbox predictions.
        """
        activation = nn.ReLU()
        """ your code starts here """
        gn = nn.GroupNorm(32, 256)
        # Fanchen Bu: There are GN layers in original FCOS
        self.shared_convs = None

        # cls_convs: [H*W*256 --> H*W*256] * 4 + [H*W*256 --> H*W*C(cls) / H*W*1(ctns)]
        self.cls_convs = nn.ModuleList(
            [Conv2d(256, 256, kernel_size=3, padding=1, norm=gn, activation=activation)
             for _ in range(self.num_stacked_convs)]
        )

        # reg_convs: [H*W*256 --> H*W*256] * 4 + [H*W*256 --> H*W*4]
        self.reg_convs = nn.ModuleList(
            [Conv2d(256, 256, kernel_size=3, padding=1, norm=gn, activation=activation)
             for _ in range(self.num_stacked_convs)]
        )

        self.cls_logits = nn.ModuleList([Conv2d(256, self.num_classes, kernel_size=3, padding=1)])
        self.bbox_pred = nn.ModuleList([Conv2d(256, 4, kernel_size=3, padding=1)])
        self.centerness = nn.ModuleList([Conv2d(256, 1, kernel_size=3, padding=1)])

        self.scales = None
        """ your code ends here """

    def _init_weights(self):
        for modules in [
            self.shared_convs, self.cls_convs, self.reg_convs,
            self.cls_logits, self.bbox_pred, self.centerness
        ]:
            if modules:
                # weight initialization with mean=0, std=0.01
                for module in modules.children():
                    normal_init(module, mean=0, std=0.01)

        # initialize the bias for classification logits
        bias_cls = -math.log((1.0 - self.prior_prob) / self.prior_prob)
        # Fanchen Bu: from the paper of RetinaNet, b = -log((1-pi)/pi)
        # calculate proper value that makes cls_probability with `self.prior_prob`
        # In other words, make the initial 'sigmoid' activation of cls_logits as `self.prior_prob`
        # by controlling bias initialization
        nn.init.constant_(self.cls_logits[0].bias, bias_cls)

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

        # self.in_features = _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
        # features = [features[f] for f in self.in_features]
        # self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
        for feat_level, feature in enumerate(features):
            """ your code starts here """
            print(feat_level, feature)
            # cls = feature
            # for layer in self.cls_convs:
            #     cls = layer(cls)
            # cls = self.cls_logits[0](cls)
            #
            # reg = feature
            # for layer in self.reg_convs:
            #     reg = layer(reg)
            # reg = self.bbox_pred[0](reg)




            """ your code ends here """
        exit(-99)
        return cls_scores, bbox_preds, centernesses
