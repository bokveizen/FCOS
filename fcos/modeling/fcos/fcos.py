import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances
from fcos.layers import IOULoss, ml_nms
from typing import Dict

from .fcos_head import FCOSHead
from .fcos_losses import FCOSLosses
from .fcos_targets import FCOSTargets, get_points

__all__ = ["FCOS"]

INF = 100000000

"""
Shape shorthand in this module:
    N: number of images in the minibatch.
    Hi, Wi: height and width of the i-th level feature map.
    4: size of the box parameterization.
Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the
        ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets
    ctrness_pred: predicted centerness scores
"""


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.normalize_reg_targets = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS
        # inference parameters
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.nms_pre_topk = cfg.MODEL.FCOS.NMS_PRE_TOPK
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_post_topk = cfg.MODEL.FCOS.NMS_POST_TOPK

        # Fanchen:
        # # Inference parameters
        # _C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
        # _C.MODEL.FCOS.NMS_THRESH_TEST = 0.6
        # _C.MODEL.FCOS.NMS_PRE_TOPK = 1000
        # _C.MODEL.FCOS.NMS_POST_TOPK = 100

        # fmt: on
        self.cfg = cfg
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

        reg_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        self.reg_loss = IOULoss(reg_loss_type)

    def forward(self, images, features, gt_instances):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "pred_boxes", "scores", "pred_classes",
            "locations".
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        # Step 1. FCOS head implementation
        fcos_preds = self.fcos_head(features)
        all_level_points = get_points(features, self.fpn_strides)

        # Fanchen: DEBUG
        # import joblib
        # for _ in range(10):
        #     print('DEBUG!!!')
        # # print(gt_instances[0])
        # joblib.dump(gt_instances[0], '/home/CtrlDrive/fanchen/pyws/ee898_pa1/gtins.data')
        # exit(0)

        # Fanchen: An example of gt_instances
        # Instances(num_instances=2, image_height=800, image_width=1196, fields=[gt_boxes: Boxes(tensor([[ 634.6836,  260.2990, 1120.6893,  791.2336],
        # [ 716.8525,  441.9065,  739.6699,  470.2243]], device='cuda:3')), gt_classes: tensor([ 0, 67], device='cuda:3')])

        if self.training:
            # Step 2. training target generation
            training_targets = FCOSTargets(all_level_points, gt_instances, self.cfg)

            # Step 3. loss computation
            loss_inputs = fcos_preds + training_targets
            losses = FCOSLosses(*loss_inputs, self.reg_loss, self.cfg)
            if self.mask_on:  # Proposal generation for Instance Segmentation (ExtraExtra)
                # compute proposals for ROI sampling
                proposals = self.predict_proposals(
                    *fcos_preds,
                    all_level_points,
                    images.image_sizes,
                )
                return proposals, losses
            else:
                return None, losses

        # Step 4. Inference phase
        proposals = self.predict_proposals(*fcos_preds, all_level_points, images.image_sizes)
        return proposals, None

    def predict_proposals(
            self,
            cls_scores,
            bbox_preds,
            centernesses,
            all_level_points,
            image_sizes
    ):
        # Fanchen: DEBUG
        # for _ in range(20):
        #     print('predict_proposals @ fcos.py now!!!!!!!!!!')
        """
        Arguments:
            cls_scores, bbox_preds, centernesses: Same as the output of :meth:`FCOSHead.forward`
            all_level_points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        num_imgs = len(image_sizes)
        num_levels = len(cls_scores)

        # recall that during training, we normalize regression targets with FPN's stride.
        # we denormalize them here.
        if self.normalize_reg_targets:
            bbox_preds = [bbox_preds[i] * self.fpn_strides[i] for i in range(num_levels)]

        result_list = []
        for img_id in range(num_imgs):
            # each entry of list corresponds to per-level feature tensor of single image.
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]

            # per-image proposal comutation
            det_bboxes = self.predict_proposals_single_image(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                all_level_points,
                image_sizes[img_id]
            )
            result_list.append(det_bboxes)
        return result_list

    def predict_proposals_single_image(
            self,
            cls_scores,
            bbox_preds,
            centernesses,
            all_level_points,
            image_size
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            cls_scores (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (C, Hi, Wi), where i denotes a specific feature level.
            bbox_preds (list[Tensor]): Same shape as 'cls_scores' except that C becomes 4.
            centernesses (list[Tensor]): Same shape as 'cls_scores' except that C becomes 1.
            all_level_points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `predict_proposals`, but for only one image.
        """
        assert len(cls_scores) == len(bbox_preds) == len(all_level_points)
        bboxes_list = []

        # Iterate over every feature level
        for (cls_score, bbox_pred, centerness, points) in zip(
                cls_scores, bbox_preds, centernesses, all_level_points
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # (C, Hi, Wi) -> (Hi*Wi, C)
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            # (4, Hi, Wi) -> (Hi*Wi, 4)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # (1, Hi, Wi) -> (Hi*Wi, )
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            # Fanchen: DEBUG
            # torch.save((cls_scores,
            #             bbox_preds,
            #             centernesses,
            #             all_level_points,
            #             image_size,
            #             scores,
            #             bbox_pred,
            #             centerness), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/inf.data')
            # print('DEBUG: inf.data')
            # exit(0)
            # >>> len(cls_scores)
            # 5
            # >>> [score.size() for score in cls_scores]
            # [torch.Size([80, 152, 100]), torch.Size([80, 76, 50]), torch.Size([80, 38, 25]), torch.Size([80, 19, 13]), torch.Size([80, 10, 7])]
            # >>> scores
            # tensor([[0.0082, 0.0043, 0.0070,  ..., 0.0048, 0.0050, 0.0046],
            #         [0.0034, 0.0016, 0.0029,  ..., 0.0021, 0.0017, 0.0015],
            #         [0.0024, 0.0013, 0.0020,  ..., 0.0018, 0.0017, 0.0013],
            #         ...,
            #         [0.0050, 0.0022, 0.0024,  ..., 0.0010, 0.0013, 0.0008],
            #         [0.0057, 0.0027, 0.0032,  ..., 0.0014, 0.0015, 0.0010],
            #         [0.0129, 0.0077, 0.0085,  ..., 0.0048, 0.0057, 0.0040]],
            #        device='cuda:7')
            # >>> scores.size()
            # torch.Size([15200, 80])
            # >>> bbox_pred, bbox_pred.size()
            # (tensor([[ 6.7271,  6.7130, 16.7200, 13.4471],
            #         [12.8911,  5.4016, 11.4462, 10.5563],
            #         [17.2124,  5.3992, 17.0486, 10.5352],
            #         ...,
            #         [22.8796, 15.5267, 19.0822,  8.6359],
            #         [28.2969, 15.3834, 15.9940,  9.6031],
            #         [18.1814, 19.1390, 12.2707, 13.9811]], device='cuda:7'), torch.Size([15200, 4]))
            # >>> centerness, centerness.size()
            # (tensor([0.1976, 0.2229, 0.2007,  ..., 0.2555, 0.2092, 0.2774], device='cuda:7'), torch.Size([15200]))
            # >>> all_level_points[0].size()
            # torch.Size([15200, 2])

            """ Your code starts here """
            # H, W = image_size
            scores_i_th_inds = torch.zeros_like(scores) + (scores > self.score_threshold)
            scores *= scores_i_th_inds
            scores *= centerness[:, None]
            topk_cnt = scores_i_th_inds.reshape(-1).sum().clamp(max=self.nms_pre_topk)
            # Fanchen: recover ltrb form, modified @ 0502
            bbox_pred = torch.stack([points[:, 0] - bbox_pred[:, 0],
                                     points[:, 1] - bbox_pred[:, 1],
                                     points[:, 0] + bbox_pred[:, 2],
                                     points[:, 1] + bbox_pred[:, 3]], dim=1)

            bbox_pred = torch.stack([points[:, 0] - bbox_pred[:, 0],
                                     points[:, 1] - bbox_pred[:, 1],
                                     points[:, 0] + bbox_pred[:, 2],
                                     points[:, 1] + bbox_pred[:, 3]], dim=1)

            flatten_scores = scores.reshape(-1)  # Fanchen: size is (H*W*C, )
            # flatten_labels = torch.tensor(range(self.num_classes)). \
            #     repeat(image_size[0] * image_size[1])  # Fanchen: size is (H*W*C, )
            flatten_boxes = bbox_pred.unsqueeze(1). \
                expand(-1, self.num_classes, -1).reshape(-1, 4)  # Fanchen: size is (H*W*C, 4)
            pred_scores, topk_inds = flatten_scores.topk(int(topk_cnt))
            pred_scores = torch.sqrt(pred_scores)
            pred_boxes = Boxes(flatten_boxes[topk_inds])
            pred_classes = topk_inds % self.num_classes
            box_list = Instances(image_size, pred_boxes=pred_boxes, scores=pred_scores, pred_classes=pred_classes)
            bboxes_list.append(box_list)
            # Fanchen: tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
            """ Your code ends here """

        bboxes_list = Instances.cat(bboxes_list)
        # Fanchen: def cat(instance_lists: List["Instances"]) -> "Instances":

        # non-maximum suppression per-image.
        results = ml_nms(
            bboxes_list,
            # Fanchen:
            # boxes = boxlist.pred_boxes.tensor
            # scores = boxlist.scores
            # labels = boxlist.pred_classes
            self.nms_threshold,
            # Limit to max_per_image detections **over all classes**
            max_proposals=self.nms_post_topk
        )
        # Fanchen: DEBUG
        # torch.save((bboxes_list, results), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/infres.data')
        # exit(0)
        return results
