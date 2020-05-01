import torch
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import cat

from .fcos_targets import compute_centerness_targets


def FCOSLosses(
        cls_scores,
        bbox_preds,
        centernesses,
        labels,
        bbox_targets,
        reg_loss,
        cfg
):
    """
    Arguments:
        cls_scores, bbox_preds, centernesses: Same as the output of :meth:`FCOSHead.forward`
        labels, bbox_targets: Same as the output of :func:`FCOSTargets`

    Returns:
        losses (dict[str: Tensor]): A dict mapping from loss name to loss value.
    """
    # fmt: off
    num_classes = cfg.MODEL.FCOS.NUM_CLASSES
    focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
    focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
    # fmt: on

    # Collect all logits and regression predictions over feature maps
    # and images to arrive at the same shape as the labels and targets
    # The final ordering is L, N, H, W from slowest to fastest axis.
    flatten_cls_scores = cat(
        [
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            cls_score.permute(0, 2, 3, 1).reshape(-1, num_classes)
            for cls_score in cls_scores
        ], dim=0)

    flatten_bbox_preds = cat(
        [
            # Reshape: (N, 4, Hi, Wi) -> (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ], dim=0)
    flatten_centernesses = cat(
        [
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            centerness.reshape(-1) for centerness in centernesses
        ], dim=0)

    # flatten classification and regression targets.
    flatten_labels = cat(labels).squeeze(1)  # Fanchen: modified
    flatten_bbox_targets = cat(bbox_targets)

    # Fanchen: DEBUG
    # torch.save((
    #     cls_scores,
    #     bbox_preds,
    #     centernesses,
    #     labels,
    #     bbox_targets,
    #     reg_loss,
    #     num_classes,
    #     focal_loss_alpha,
    #     focal_loss_gamma,
    #     flatten_cls_scores,
    #     flatten_bbox_preds,
    #     flatten_centernesses,
    #     flatten_labels,
    #     flatten_bbox_targets), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/loss.data')
    # exit(0)

    # retain indices of positive predictions.
    pos_inds = torch.nonzero(flatten_labels != num_classes).squeeze(1)
    num_pos = max(len(pos_inds), 1.0)

    # prepare one_hot label.
    class_target = torch.zeros_like(flatten_cls_scores)
    # Fanchen: DEBUG
    # torch.save((flatten_cls_scores, pos_inds, class_target, flatten_labels), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/clstgt.data')
    # exit(0)
    # >>> flatten_cls_scores, flatten_cls_scores.size()
    # (tensor([[-4.5902, -4.5959, -4.5890,  ..., -4.5900, -4.5927, -4.5966],
    #         [-4.5862, -4.5947, -4.5896,  ..., -4.5841, -4.5924, -4.5890],
    #         [-4.5939, -4.5958, -4.5867,  ..., -4.5892, -4.5873, -4.5959],
    #         ...,
    #         [-4.5891, -4.5946, -4.5883,  ..., -4.5964, -4.5975, -4.5964],
    #         [-4.5902, -4.5981, -4.5899,  ..., -4.5947, -4.5974, -4.6001],
    #         [-4.5937, -4.5950, -4.5929,  ..., -4.5948, -4.5974, -4.5986]],
    #        device='cuda:0', requires_grad=True), torch.Size([89600, 80]))
    # >>> pos_inds, pos_inds.size()
    # (tensor([[ 3196,     0],
    #         [ 3197,     0],
    #         [ 3198,     0],
    #         ...,
    #         [89563,     0],
    #         [89569,     0],
    #         [89573,     0]], device='cuda:0'), torch.Size([217, 2]))
    # >>> class_target, class_target.size()
    # (tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         ...,
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.],
    #         [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0'), torch.Size([89600, 80]))
    # >>> flatten_labels.size()
    # torch.Size([89600, 1])
    # >>> flatten_labels[pos_inds].size()
    # torch.Size([217, 2, 1])
    # >>> class_target[pos_inds].size()
    # torch.Size([217, 2, 80])
    # >>> class_target[pos_inds, flatten_labels[pos_inds]] = 1
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes [217, 2], [217, 2, 1]
    # class_target[pos_inds[:, 0], flatten_labels[pos_inds[:, 0]]] = 1  # Fanchen: try to fix
    class_target[pos_inds, flatten_labels[pos_inds]] = 1  # original

    # classification loss: Focal loss
    loss_cls = sigmoid_focal_loss_jit(
        flatten_cls_scores,
        class_target,
        alpha=focal_loss_alpha,
        gamma=focal_loss_gamma,
        reduction="sum",
    ) / num_pos

    # compute regression loss and centerness loss only for positive samples.
    # pos_bbox_preds = flatten_bbox_preds[pos_inds[:, 0]]  # Fanchen: try to fix
    # pos_centernesses = flatten_centernesses[pos_inds[:, 0]]  # Fanchen: try to fix
    # pos_bbox_targets = flatten_bbox_targets[pos_inds[:, 0]]  # Fanchen: try to fix
    pos_bbox_preds = flatten_bbox_preds[pos_inds]
    pos_centernesses = flatten_centernesses[pos_inds]
    pos_bbox_targets = flatten_bbox_targets[pos_inds]

    # Fanchen: DEBUG
    # torch.save((flatten_bbox_preds, pos_inds, pos_bbox_preds, pos_centernesses, pos_bbox_targets),
    #            '/home/CtrlDrive/fanchen/pyws/ee898_pa1/pos.data')
    # exit(0)
    # >>> pos_bbox_preds, pos_bbox_preds.size()
    # (tensor([[[0.9752, 0.9950, 1.0029, 1.0040],
    #          [0.9935, 0.9977, 0.9974, 1.0071]],
    #
    #         [[0.9810, 0.9901, 1.0076, 1.0037],
    #          [0.9935, 0.9977, 0.9974, 1.0071]],
    #
    #         [[0.9909, 0.9973, 0.9954, 1.0087],
    #          [0.9935, 0.9977, 0.9974, 1.0071]],
    #
    #         ...,
    #
    #         [[0.9940, 0.9997, 0.9920, 0.9980],
    #          [0.9935, 0.9977, 0.9974, 1.0071]],
    #
    #         [[0.9930, 0.9961, 0.9954, 0.9947],
    #          [0.9935, 0.9977, 0.9974, 1.0071]],
    #
    #         [[0.9987, 0.9956, 0.9966, 0.9980],
    #          [0.9935, 0.9977, 0.9974, 1.0071]]], device='cuda:0',
    #        requires_grad=True), torch.Size([1174, 2, 4]))
    # >>> pos_centernesses, pos_centernesses.size()
    # (tensor([-0.0043,  0.0030,  0.0037,  ...,  0.0037, -0.0006, -0.0025],
    #        device='cuda:0', requires_grad=True), torch.Size([81068]))
    # >>> pos_bbox_targets, pos_bbox_targets.size()
    # (tensor([[ -8.7904, -41.2979,  17.1076,  54.1063],
    #         [ -7.7904, -41.2979,  16.1076,  54.1063],
    #         [ -6.7904, -41.2979,  15.1076,  54.1063],
    #         ...,
    #         [  4.0954,   3.6686,  -3.5032,  -3.3509],
    #         [  5.0954,   3.6686,  -4.5032,  -3.3509],
    #         [  6.0954,   3.6686,  -5.5032,  -3.3509]], device='cuda:0'), torch.Size([81068, 4]))

    # compute centerness targets.
    pos_centerness_targets = compute_centerness_targets(pos_bbox_targets)
    centerness_norm = max(pos_centerness_targets.sum(), 1e-6)

    # regression loss: IoU loss
    loss_bbox = reg_loss(
        pos_bbox_preds,
        pos_bbox_targets,
        weight=pos_centerness_targets
    ) / centerness_norm

    # centerness loss: Binary CrossEntropy loss
    loss_centerness = F.binary_cross_entropy_with_logits(
        pos_centernesses,
        pos_centerness_targets,
        reduction="sum"
    ) / num_pos

    # final loss dict.
    losses = dict(
        loss_fcos_cls=loss_cls,
        loss_fcos_loc=loss_bbox,
        loss_fcos_ctr=loss_centerness
    )
    return losses
