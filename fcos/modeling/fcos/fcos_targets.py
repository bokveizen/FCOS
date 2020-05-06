import torch

from detectron2.layers import cat
from fcos.utils import multi_apply

INF = 100000000


def FCOSTargets(all_level_points, gt_instances, cfg):
    # fmt: off
    num_classes = cfg.MODEL.FCOS.NUM_CLASSES
    fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
    sizes_of_interest = cfg.MODEL.FCOS.SIZES_OF_INTEREST
    center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
    center_radius = cfg.MODEL.FCOS.POS_RADIUS
    normalize_reg_targets = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS
    # fmt: on

    regress_ranges = generate_regress_ranges(sizes_of_interest)

    center_sample_cfg = dict(center_sample=center_sample, center_radius=center_radius)

    return fcos_target(
        all_level_points,
        regress_ranges,
        gt_instances,
        fpn_strides,
        center_sample_cfg,
        normalize_reg_targets,
        num_classes=num_classes
    )


def generate_regress_ranges(sizes_of_interest):
    # generate sizes of interest
    regress_ranges = []
    prev_size = -1
    for s in sizes_of_interest:
        regress_ranges.append([prev_size, s])
        prev_size = s
    regress_ranges.append([prev_size, INF])
    return regress_ranges


def get_points(features, fpn_strides):
    """Get points according to feature map sizes.

    Args:
        features (list[Tensor]): Multi-level feature map. Axis 0 represents the number of
            images `N` in the input data; axes 1-3 are channels, height, and width, which
            may vary between feature maps (e.g., if a feature pyramid is used).
        fpn_strides (list[int]): Feature map strides corresponding to each level of multi-level
            feature map.

    Returns:
        points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
    """
    assert len(features) == len(fpn_strides)

    points = []
    for feat, stride in zip(features, fpn_strides):
        featmap_size = feat.size()[-2:]
        points.append(
            # run on single feature-level
            get_points_single(featmap_size, stride, feat.device))
    # Fanchen: DEBUG
    # torch.save((features, fpn_strides, points), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/getpoints.data')
    # exit(0)
    # >>> len(features)
    # 5
    # >>> [feature.size() for feature in features]
    # [torch.Size([2, 256, 100, 152]), torch.Size([2, 256, 50, 76]), torch.Size([2, 256, 25, 38]), torch.Size([2, 256, 13, 19]), torch.Size([2, 256, 7, 10])]
    # >>> fpn_strides
    # [8, 16, 32, 64, 128]
    # >>> len(points)
    # 5
    # >>> [point.size() for point in points]
    # [torch.Size([15200, 2]), torch.Size([3800, 2]), torch.Size([950, 2]), torch.Size([247, 2]), torch.Size([70, 2])]
    return points


def get_points_single(featmap_size, stride, device):
    """point prediction per feature-level.
    # Fanchen: stride should be int
    Args:
        featmap_size (Tuple): feature map size (Hi, Wi) where 'i' denotes specific feature level.
        stride (int): feature map stride corresponding to each feature level 'i'.
        device: the same device type with feature map tensor.

    Returns:
        points (Tensor): Tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi)
            of all feature map locations on feature-level 'i' in image coordinate.
    """

    """ your code starts here """

    # Fanchen: Example of meshgrid
    # >>> x = torch.tensor([1, 2, 3])
    # >>> y = torch.tensor([4, 5, 6])
    # >>> grid_x, grid_y = torch.meshgrid(x, y)
    # >>> grid_x
    # tensor([[1, 1, 1],
    #         [2, 2, 2],
    #         [3, 3, 3]])
    # >>> grid_y
    # tensor([[4, 5, 6],
    #         [4, 5, 6],
    #         [4, 5, 6]])

    h, w = featmap_size
    x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    # Fanchen: x = [0, s, 2s,..., (w-1) * s], y = [0, s, 2s,..., (h-1) * s]
    grid_y, grid_x = torch.meshgrid(y, x)
    # Fanchen: grid_x = [[0, s, 2s,..., (w-1) * s] * h]
    # grid_y = [[0 * w], [s * w], [2s * w],...,[{(h-1) * s} * w]]
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1) + stride // 2
    # points.shape -> torch.Size([35, 2]) (when w = 5, h = 7)
    """ your code ends here """
    return points


def fcos_target(
        points,
        regress_ranges,
        gt_instance_list,
        fpn_strides,
        center_sample_cfg,
        normalize_reg_targets,
        num_classes=80
):
    """Compute class labels and regression targets for every feature points on all feature levels.

    Args:
        points (list[Tensor]): list of #feature levels.
            Each entry contains tensor of size (N*Hi*Wi, )
        regress_ranges (list[tuple]): list of #feature levels. Each entry denotes the
            lower bound and upper bound of regression range of bbox target
            for the corresponding feature level.
        gt_instance_list (list[Instances]): a length `N` list of `Instances`s.
            Each `Instances` stores ground-truth instances for the corresponding image.
        fpn_strides (list[int]): list of #feature levels.
        center_sample_cfg (dict): hyperparameters for center sampling.
        normalize_reg_targets (bool): whether to normalize regression targets by each stride of
            corresponding feature stride.
        num_classes (int)

    Returns:
        concat_labels (list[Tensor]): list of #feature levels. Each entry contains
            tensor of size (N*Hi*Wi, )
        concat_bbox_targets (list[Tensor]): list of #feature levels. Each entry contains
            tensor of size (N*Hi*Wi, 4)
    """
    assert len(points) == len(regress_ranges)
    num_levels = len(points)

    # expand regress ranges to align with points
    expanded_regress_ranges = [
        points[i].new_tensor(regress_ranges[i])[None].expand_as(points[i])
        for i in range(num_levels)
    ]

    # concat all levels points and regress ranges
    concat_regress_ranges = cat(expanded_regress_ranges, dim=0)
    concat_points = cat(points, dim=0)

    # the number of points per img, per level
    num_points = [center.size(0) for center in points]

    # get labels and bbox_targets of each image; per-image computation.

    # Fanchen: def of multi_apply
    # def multi_apply(func, *args, **kwargs):
    #     pfunc = partial(func, **kwargs) if kwargs else func
    #     # Fanchen: partial, pfunc = func w/ **kwargs fixed as **kwargs
    #     map_results = map(pfunc, *args)  # Fanchen: return the list [pfunc(arg) for arg in *args]
    #     return tuple(map(list, zip(*map_results)))
    #     # Fanchen: map_results = [[1,2,3],[4,5,6]]; tuple(map(list, zip(*map_results)))
    #     # ([1, 4], [2, 5], [3, 6])

    labels_list, bbox_targets_list = multi_apply(
        fcos_target_single_image,
        gt_instance_list,
        points=concat_points,
        regress_ranges=concat_regress_ranges,
        num_points_per_level=num_points,
        fpn_strides=fpn_strides,
        center_sample_cfg=center_sample_cfg,
        normalize_reg_targets=normalize_reg_targets,
        num_classes=num_classes
    )
    # Fanchen: DEBUG
    # torch.save((labels_list, bbox_targets_list, num_points), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/tgt2.data')
    # exit(0)
    # split to per img, per feature level
    labels_list = [labels.split(num_points, 0) for labels in labels_list]
    bbox_targets_list = [
        bbox_targets.split(num_points, 0)
        for bbox_targets in bbox_targets_list
    ]
    # >>> [kk.size() for kk in ll[0]]
    # [torch.Size([16400, 1]), torch.Size([4100, 1]), torch.Size([1025, 1]), torch.Size([273, 1]), torch.Size([77, 1])]
    # >>> [kk.size() for kk in btl[0]]
    # [torch.Size([16400, 4]), torch.Size([4100, 4]), torch.Size([1025, 4]), torch.Size([273, 4]), torch.Size([77, 4])]
    # >>> num_points
    # [16400, 4100, 1025, 273, 77]

    # concat per level image
    concat_labels = []
    concat_bbox_targets = []
    for i in range(num_levels):
        concat_labels.append(
            cat([labels[i] for labels in labels_list])
        )

        if normalize_reg_targets:
            # we normalize reg_targets by FPN's strides here
            normalizer = float(fpn_strides[i])
        else:
            normalizer = 1.0

        concat_bbox_targets.append(
            cat([bbox_targets[i] / normalizer for bbox_targets in bbox_targets_list])
        )
    # Fanchen: DEBUG
    # torch.save((points,
    #             regress_ranges,
    #             gt_instance_list,
    #             fpn_strides,
    #             center_sample_cfg,
    #             normalize_reg_targets,
    #             num_classes,
    #             concat_labels,
    #             concat_bbox_targets), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/fcostgt.data')
    # exit(0)
    # >>> len(points)
    # 5
    # >>> [point.size() for point in points]
    # [torch.Size([16800, 2]), torch.Size([4200, 2]), torch.Size([1050, 2]), torch.Size([273, 2]), torch.Size([77, 2])]
    # >>> [concat_label.size() for concat_label in concat_labels]
    # [torch.Size([33600, 1]), torch.Size([8400, 1]), torch.Size([2100, 1]), torch.Size([546, 1]), torch.Size([154, 1])]
    # >>> [tgt.size() for tgt in concat_bbox_targets]
    # [torch.Size([33600, 4]), torch.Size([8400, 4]), torch.Size([2100, 4]), torch.Size([546, 4]), torch.Size([154, 4])]

    return concat_labels, concat_bbox_targets


def fcos_target_single_image(
        gt_instances,
        points,
        regress_ranges,
        num_points_per_level,
        fpn_strides,
        center_sample_cfg,
        normalize_reg_targets,
        num_classes=80
):
    """Compute class labels and regression targets for single image.

    Args:
        gt_instances (Instances): stores ground-truth instances for the corresponding image.
        all other args are the same as in `self.fcos_target` where all elements in the list
            are concatenated to form a single tensor.

    Returns:
        labels (Tensor): class label of every feature point in all feature levels for single image.
        bbox_targets (Tensor): regression targets of every feature point in all feature levels
            for a single image. each column corresponds to a tensor shape of (l, t, r, b).
    """
    center_sample = center_sample_cfg['center_sample']
    center_radius = center_sample_cfg['center_radius']

    # Fanchen: DEBUG
    # for _ in range(5):
    #     print('DEBUG: fcos_target_single_image')
    # torch.save((
    #     gt_instances,
    #     points,
    #     regress_ranges,
    #     num_points_per_level,
    #     fpn_strides,
    #     center_sample_cfg,
    #     normalize_reg_targets,
    #     num_classes,
    #     center_sample,
    #     center_radius
    # ), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/tgt.data')
    # exit(0)

    # >>> gt_instances
    # Instances(num_instances=2, image_height=800, image_width=1199,
    # fields=[gt_boxes: Boxes(tensor([[ 115.5350,   30.2576,  803.5361,  732.7401],
    # [ 535.6720,   14.3888, 1199.0000,  780.2155]], device='cuda:3')),
    # gt_classes: tensor([22, 22], device='cuda:3')])

    # >>> points, points.size()
    # (tensor([[   4.,    4.],
    #         [  12.,    4.],
    #         [  20.,    4.],
    #         ...,
    #         [ 960.,  832.],
    #         [1088.,  832.],
    #         [1216.,  832.]], device='cuda:3'), torch.Size([20267, 2]))

    # >>> regress_ranges, regress_ranges.size()
    # (tensor([[-1.0000e+00,  6.4000e+01],
    #         [-1.0000e+00,  6.4000e+01],
    #         [-1.0000e+00,  6.4000e+01],
    #         ...,
    #         [ 5.1200e+02,  1.0000e+08],
    #         [ 5.1200e+02,  1.0000e+08],
    #         [ 5.1200e+02,  1.0000e+08]], device='cuda:3'), torch.Size([20267, 2]))

    # >>> num_points_per_level
    # [15200, 3800, 950, 247, 70]
    # >>> fpn_strides
    # [8, 16, 32, 64, 128]
    # >>> center_sample_cfg
    # {}
    # >>> normalize_reg_targets
    # True
    # >>> num_classes
    # 80
    # >>> center_sample
    # False
    # >>> center_radius
    # 1.5

    # here, num_points accumulates all locations across all feature levels.
    num_points = points.size(0)  # Fanchen: 20267, Sum of W*H for all feature levels, len(all sampling positions)
    num_gts = len(gt_instances)  # Fanchen: 2, The # of bboxes in the img

    # Fanchen: An example of gt_instances (gt_instaces[0]) Instances(num_instances=2, image_height=800,
    # image_width=1196, fields=[gt_boxes: Boxes(tensor([[ 634.6836,  260.2990, 1120.6893,  791.2336], [ 716.8525,
    # 441.9065,  739.6699,  470.2243]], device='cuda:3')), gt_classes: tensor([ 0, 67], device='cuda:3')])

    # Fanchen: Boxes --> detectron2.structures.boxes
    # Fanchen: DEBUG
    # torch.save(gt_instances, '/home/CtrlDrive/fanchen/pyws/ee898_pa1/gt_inst.data')
    # exit(-1)

    # get class labels and bboxes from `gt_instances`.
    gt_labels = gt_instances.gt_classes
    # Fanchen: >>> gt_instances.gt_classes
    # tensor([22, 22], device='cuda:3')
    gt_bboxes = gt_instances.gt_boxes
    # Fanchen: >>> gt_bboxes
    # Boxes(tensor([[ 115.5350,   30.2576,  803.5361,  732.7401],
    #         [ 535.6720,   14.3888, 1199.0000,  780.2155]], device='cuda:3'))

    if num_gts == 0:
        return (
            gt_labels.new_zeros(num_points) + num_classes,
            gt_bboxes.new_zeros((num_points, 4))
        )

    # `areas`: should be `torch.Tensor` shape of (num_points, num_gts, 1)

    # Fanchen: TEST
    # num_gts, num_points = 5, 2
    # areas = torch.zeros(num_gts, 1)  # 1. `torch.Tensor` shape of (num_gts, 1)
    # areas = areas[None].repeat(num_points, 1, 1)  # 2. hint: use :func:`torch.repeat`.
    # areas.size() -> torch.Size([2, 5, 1])

    # areas = gt_bboxes.area().view(-1, 1)  # 1. `torch.Tensor` shape of (num_gts, 1)
    areas = gt_bboxes.area().reshape(-1, 1)  # 1. `torch.Tensor` shape of (num_gts, 1)
    # Fanchen: >>> gt_bboxes.area().view(-1, 1)
    # tensor([[483308.6875],
    #         [507994.3125]], device='cuda:3')
    # >>> gt_bboxes.area().view(-1, 1).size()
    # torch.Size([2, 1])
    areas = areas[None].repeat(num_points, 1, 1)  # 2. hint: use :func:`torch.repeat`.
    # Fanchen: >>> areas
    # tensor([[[483308.6875],
    #          [507994.3125]],
    #
    #         [[483308.6875],
    #          [507994.3125]],
    #
    #         [[483308.6875],
    #          [507994.3125]],
    #
    #         ...,
    #
    #         [[483308.6875],
    #          [507994.3125]],
    #
    #         [[483308.6875],
    #          [507994.3125]],
    #
    #         [[483308.6875],
    #          [507994.3125]]], device='cuda:3')
    # >>> areas.size()
    # torch.Size([20267, 2, 1])

    # Fanchen: vals, indices = areas.min(dim=1)

    # Fanchen: DEBUG
    # print(regress_ranges.size(), num_points, num_gts)
    # exit(0)
    # torch.Size([22400, 2]) 22400 11
    # torch.Size([20267, 2]) 20267 1
    # torch.Size([20267, 2]) 20267 13
    # torch.Size([20267, 2]) 20267 16

    # `regress_ranges`: should be `torch.Tensor` shape of (num_points, num_gts, 2)
    # regress_ranges = cat([r.expand(num_gts, 2) for r in regress_ranges]).view(num_points, num_gts, 2)
    # regress_ranges = cat([r.expand(num_gts, 2) for r in regress_ranges]).reshape(num_points, num_gts, 2)
    regress_ranges = regress_ranges.unsqueeze(1).expand(num_points, num_gts, 2)  # Fanchen: 0501, modified
    # hint: use :func:`torch.expand`.
    # Fanchen: >>> regress_ranges
    # tensor([[[-1.0000e+00,  6.4000e+01],
    #          [-1.0000e+00,  6.4000e+01]],
    #
    #         [[-1.0000e+00,  6.4000e+01],
    #          [-1.0000e+00,  6.4000e+01]],
    #
    #         [[-1.0000e+00,  6.4000e+01],
    #          [-1.0000e+00,  6.4000e+01]],
    #
    #         ...,
    #
    #         [[ 5.1200e+02,  1.0000e+08],
    #          [ 5.1200e+02,  1.0000e+08]],
    #
    #         [[ 5.1200e+02,  1.0000e+08],
    #          [ 5.1200e+02,  1.0000e+08]],
    #
    #         [[ 5.1200e+02,  1.0000e+08],
    #          [ 5.1200e+02,  1.0000e+08]]], device='cuda:3')
    # >>> regress_ranges.size()
    # torch.Size([20267, 2, 2])

    # `gt_bboxes`: should be `torch.Tensor` shape of (num_points, num_gts, 4)
    # Fanchen: gt_bboxes.tensor.size() -> [num_gts, 4]
    gt_bboxes = gt_bboxes.tensor.expand(num_points, num_gts, 4)  # hint: use :func:`torch.expand`.
    # Fanchen: >>> gt_bboxes
    # tensor([[[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]],
    #
    #         [[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]],
    #
    #         [[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]],
    #
    #         ...,
    #
    #         [[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]],
    #
    #         [[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]],
    #
    #         [[ 115.5350,   30.2576,  803.5361,  732.7401],
    #          [ 535.6720,   14.3888, 1199.0000,  780.2155]]], device='cuda:3')
    # >>> gt_bboxes.size()
    # torch.Size([20267, 2, 4])

    # align each coordinate  component xs, ys in shape as (num_points, num_gts)
    xs, ys = points[:, 0], points[:, 1]  # Fanchen: xs.size(), ys.size() -> [num_points, ]
    # xs = xs.view(-1, 1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.
    # ys = ys.view(-1, 1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.
    xs = xs.reshape(-1, 1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.
    ys = ys.reshape(-1, 1).expand(num_points, num_gts)  # hint: use :func:`torch.expand`.
    # Fanchen: >>> xs, xs.size()
    # (tensor([[   4.,    4.],
    #         [  12.,   12.],
    #         [  20.,   20.],
    #         ...,
    #         [ 960.,  960.],
    #         [1088., 1088.],
    #         [1216., 1216.]], device='cuda:3'), torch.Size([20267, 2]))

    # Fanchen: DEBUG
    # torch.save((xs, ys, gt_bboxes), '/home/CtrlDrive/fanchen/pyws/ee898_pa1/tt.data')
    # exit(0)
    # >>> xs.size(), ys.size(), bbs.size()
    # (torch.Size([18134, 3]), torch.Size([18134, 3]), torch.Size([18134, 3, 4]))

    # distances to each four side of gt bboxes.
    # The equations correspond to equation(1) from FCOS paper.
    left = xs[:, :] - gt_bboxes[:, :, 0]
    right = gt_bboxes[:, :, 2] - xs[:, :]
    top = ys[:, :] - gt_bboxes[:, :, 1]
    bottom = gt_bboxes[:, :, 3] - ys[:, :]
    # Fanchen: lrtb.size() -> (num_points, num_gts)
    bbox_targets = torch.stack((left, top, right, bottom), -1)  # Fanchen: w/ size [num_points, num_gts, 4]

    if center_sample:
        # This codeblock corresponds to extra credits. Note that `Not mandatory`.
        # condition1: inside a `center bbox`
        radius = center_radius
        center_xs = (gt_bboxes[:, :, 0] + gt_bboxes[:, :, 2]) * 0.5  # center x-coordinates of gt_bboxes
        center_ys = (gt_bboxes[:, :, 1] + gt_bboxes[:, :, 3]) * 0.5  # center y-coordinates of gt_bboxes
        # Fanchen: center_xs, ys w/ size [num_points, num_gts]
        center_gts = torch.zeros_like(gt_bboxes)  # Fanchen: w/ size [num_points, num_gts, 4]
        stride = center_xs.new_zeros(center_xs.shape)  # Fanchen: w/ size [num_points, num_gts]

        # project the points on current level back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_level):
            lvl_end = lvl_begin + num_points_lvl
            # radius back-projected to image coordinates
            # hint: use `fpn_strides` and `radius`
            stride[lvl_begin:lvl_end] = fpn_strides[lvl_idx] * radius
            lvl_begin = lvl_end

        # The boundary coordinates w.r.t radius(stride) and center points
        # (center coords) (- or +) (stride)
        x_mins = center_xs - stride  # Fanchen: w/ size [num_points, num_gts]
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride

        # Clip each four coordinates so that (x_mins, y_mins) and (x_maxs, y_maxs) are
        #   inside gt_bboxes. HINT: use :func:`torch.where`.
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes[:, :, 0], x_mins, gt_bboxes[:, :, 0])
        center_gts[..., 1] = torch.where(y_mins > gt_bboxes[:, :, 1], y_mins, gt_bboxes[:, :, 1])
        center_gts[..., 2] = torch.where(x_maxs < gt_bboxes[:, :, 2], x_maxs, gt_bboxes[:, :, 2])
        center_gts[..., 3] = torch.where(y_maxs < gt_bboxes[:, :, 3], y_maxs, gt_bboxes[:, :, 3])

        # distances from a location to each side of the bounding box
        # Refer to equation(1) from FCOS paper.
        # Fanchen:
        # left = xs[:, :] - gt_bboxes[:, :, 0]
        # right = gt_bboxes[:, :, 2] - xs[:, :]
        # top = ys[:, :] - gt_bboxes[:, :, 1]
        # bottom = gt_bboxes[:, :, 3] - ys[:, :]

        cb_dist_left = xs[:, :] - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs[:, :]
        cb_dist_top = ys[:, :] - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys[:, :]
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom),
            -1
        )
        # condition1: a point from center_bbox should be inside a gt bbox
        # all distances (center_l, center_t, center_r, center_b) > 0
        # hint: all distances (l, t, r, b) > 0. use :func:`torch.min`.
        inside_gt_bbox_mask = center_bbox.min(dim=-1)[0] > 0
    else:
        # condition1: a point should be inside a gt bbox
        # hint: all distances (l, t, r, b) > 0. use :func:`torch.min`.
        inside_gt_bbox_mask = bbox_targets.min(dim=-1)[0] > 0  # Fanchen: w/ size [num_points, num_gts]

    # condition2: limit the regression range for each location
    max_regress_distance = bbox_targets.max(dim=-1)[0]  # hint: use :func:`torch.max`.
    # Fanchen: torch.Size([18134, 3]) (num_points, num_gts)

    # The mask whether `max_regress_distance` on every points is bounded
    #   between the side values regress_ranges.
    # See section 3.2 3rd paragraph on FCOS paper.
    inside_regress_range = (max_regress_distance >= regress_ranges[:, :, 0]) & \
                           (max_regress_distance <= regress_ranges[:, :, 1])  # Fanchen: w/ size [num_points, num_gts]

    # filter areas that violate condition1 and condition2 above.

    # Fanchen:
    # >>> temp = torch.tensor([1,2,3])
    # >>> bool = torch.tensor([True, False, True])
    # >>> temp[bool==0]=-1
    # >>> temp
    # tensor([ 1, -1,  3])
    # torch.save(areas, '/home/CtrlDrive/fanchen/pyws/ee898_pa1/areas.data')
    # exit(0)

    areas[inside_gt_bbox_mask == 0] = INF  # use `inside_gt_bbox_mask`
    areas[inside_regress_range == 0] = INF  # use `inside_regress_range`

    # If there are still more than one objects for a location,
    # we choose the one with minimal area across `num_gts` axis.
    # Hint: use :func:`torch.min`.
    min_area, min_area_inds = areas.min(dim=1)  # Fanchen: both w/ size [num_points, 1]

    # ground-truth assignments w.r.t. bbox area indices
    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = num_classes
    # Fanchen: ERROR!
    # Traceback (most recent call last):
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
    #     fn(i, *args)
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/detectron2/engine/launch.py", line 89, in _distributed_worker
    #     main_func(*args)
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/train_net.py", line 215, in main
    #     return trainer.train()
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/train_net.py", line 95, in train
    #     self.train_loop(self.start_iter, self.max_iter)
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/train_net.py", line 85, in train_loop
    #     self.run_step()
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/detectron2/engine/train_loop.py", line 215, in run_step
    #     loss_dict = self.model(data)
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
    #     result = self.forward(*input, **kwargs)
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/torch/nn/parallel/distributed.py", line 445, in forward
    #     output = self.module(*inputs[0], **kwargs[0])
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
    #     result = self.forward(*input, **kwargs)
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/modeling/meta_arch/one_stage_detector.py", line 13, in forward
    #     return super().forward(batched_inputs)
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/detectron2/modeling/meta_arch/rcnn.py", line 245, in forward
    #     proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
    #   File "/home/CtrlDrive/fanchen/anaconda3/envs/torch/lib/python3.8/site-packages/torch/nn/modules/module.py", line 550, in __call__
    #     result = self.forward(*input, **kwargs)
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/modeling/fcos/fcos.py", line 92, in forward
    #     training_targets = FCOSTargets(all_level_points, gt_instances, self.cfg)
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/modeling/fcos/fcos_targets.py", line 23, in FCOSTargets
    #     return fcos_target(
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/modeling/fcos/fcos_targets.py", line 170, in fcos_target
    #     labels_list, bbox_targets_list = multi_apply(
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/utils/misc.py", line 8, in multi_apply
    #     return tuple(map(list, zip(*map_results)))
    #   File "/home/CtrlDrive/fanchen/pyws/ee898_pa1/fcos/modeling/fcos/fcos_targets.py", line 390, in fcos_target_single_image
    #     bbox_targets = bbox_targets[range(num_points), min_area_inds]
    # RuntimeError: CUDA out of memory. Tried to allocate 6.12 GiB (GPU 1; 11.91 GiB total capacity; 7.14 GiB already allocated; 4.18 GiB free; 7.20 GiB reserved in total by PyTorch)

    # Fanchen: DEBUG
    # torch.save((bbox_targets, num_points, min_area_inds, inside_regress_range, labels),
    #            '/home/CtrlDrive/fanchen/pyws/ee898_pa1/bbstgt.data')
    # exit(0)
    # >>> num_points
    # 20267
    # >>> bbox_targets.size()
    # torch.Size([20267, 7, 4])
    # >>> min_area_inds.size()
    # torch.Size([20267, 1])
    # >>> bbox_targets[range(num_points), min_area_inds]
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # RuntimeError: CUDA out of memory. Tried to allocate 6.12 GiB (GPU 1; 11.91 GiB total capacity; 6.12 GiB already allocated; 5.29 GiB free; 6.14 GiB reserved in total by PyTorch)
    # >>> bbox_targets[0, min_area_inds[0]]
    # tensor([[-142.7276, -279.8969,  219.0515,  695.4286]], device='cuda:1')
    # >>> bbox_targets[range(2), min_area_inds[:2]]
    # tensor([[[-142.7276, -279.8969,  219.0515,  695.4286],
    #          [-134.7276, -279.8969,  211.0515,  695.4286]],
    #
    #         [[-142.7276, -279.8969,  219.0515,  695.4286],
    #          [-134.7276, -279.8969,  211.0515,  695.4286]]], device='cuda:1')
    # torch.save((bbox_targets, num_points, min_area_inds, inside_regress_range, labels),
    #            '/home/CtrlDrive/fanchen/pyws/ee898_pa1/bbstgt0501.data')
    # exit(0)
    # Fanchen: modified
    # bbox_targets = cat([bbox_targets[i, min_area_inds[i]] for i in range(num_points)])
    bbox_targets = bbox_targets[range(num_points), min_area_inds.squeeze(1)]  # Fanchen: 0501 modified
    # bbox_targets = bbox_targets[range(num_points), min_area_inds]  # original
    # bbox_targets = bbox_targets[torch.arange(num_points), min_area_inds]
    return labels, bbox_targets


def compute_centerness_targets(pos_bbox_targets):
    """Compute centerness targets for every feature points, given bbox targets.

    Args:
        pos_bbox_targets (Tensor): regression targets of every positive feature point in all
            feature levels and for all images. Each column corresponds to a tensor shape of
            (l, t, r, b). shape of (num_pos_samples, 4)

    Returns:
        centerness_targets (Tensor): A tensor with same rows from 'pos_bbox_targets' Tensor.
    """

    """ your code starts here """
    # Fanchen: DEBUG
    # torch.save(pos_bbox_targets, '/home/CtrlDrive/fanchen/pyws/ee898_pa1/debugdata/ctnstgt.data')
    # exit(0)

    lr = pos_bbox_targets[:, [0, 2]]
    tb = pos_bbox_targets[:, [1, 3]]
    # centerness_targets = pos_bbox_targets.new_zeros(pos_bbox_targets.size(0), 1)
    # Fanchen: ctrn = sqrt( (min(lr) * min(tb)) / (max(lr) * max(tb)) )
    centerness_targets = torch.sqrt(
        (lr.min(dim=1).values * tb.min(dim=1).values) / (lr.max(dim=1).values * tb.max(dim=1).values))
    """ your code ends here """

    return centerness_targets
