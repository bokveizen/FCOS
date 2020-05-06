import cv2 as cv
import os
import torch

cat_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush']

img_path = 'D:/val2017/'
img_list = os.listdir(img_path)
pred = torch.load('instances_predictions_gn.pth')
save_path = 'vis_res/'

for i in range(len(img_list)):
    img_file = img_path + img_list[i]
    bbox_pred = [[cat_names[a['category_id']], a['bbox'], a['score']] for a in pred[i]['instances']]
    bbox_pred = sorted(bbox_pred, key=lambda x: x[-1], reverse=True)
    img = cv.imread(img_file)
    for bbox in bbox_pred[:]:
        cv.rectangle(img, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[1][0] + bbox[1][2]), int(bbox[1][1] + bbox[1][3])),
                     (255, 255, 255),
                     thickness=2)
        cv.putText(img, bbox[0] + ', ' + str(bbox[2])[:4], (int(bbox[1][0]), int(bbox[1][1])), 4, 0.4,
                   (0, 0, 255),
                   thickness=1)
        cv.imwrite(save_path + 'all/' + img_list[i], img)
    img = cv.imread(img_file)
    for bbox in bbox_pred[:10]:
        cv.rectangle(img, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[1][0] + bbox[1][2]), int(bbox[1][1] + bbox[1][3])),
                     (255, 255, 255),
                     thickness=2)
        cv.putText(img, bbox[0] + ', ' + str(bbox[2])[:4], (int(bbox[1][0]), int(bbox[1][1])), 4, 0.4,
                   (0, 0, 255),
                   thickness=1)
        cv.imwrite(save_path + 'top10/' + img_list[i], img)