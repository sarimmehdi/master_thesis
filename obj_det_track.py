import argparse
import os
import natsort
import cv2
import mmcv
import torch
import copy
import pandas as pd
import numpy as np
import pycocotools.mask as maskUtils
from math import sin, cos, degrees
from pathlib import Path
from shapely.geometry import Polygon
from DetectoRS.mmdet.apis import init_detector, inference_detector
from detectron_2.detectron2.config import get_cfg
from detectron_2.demo.predictor import VisualizationDemo

# generate evenly spaced points inside rotated rect
def generate_pts_in_rotated_rect(bbox_pts, res=0.3):
    x1, y1 = bbox_pts[0][0], bbox_pts[0][1]
    x2, y2 = bbox_pts[1][0], bbox_pts[1][1]
    x3, y3 = bbox_pts[2][0], bbox_pts[2][1]
    x4, y4 = bbox_pts[3][0], bbox_pts[3][1]

    s, t, a, b = 0, 0, 0, 0
    inter_pts = []
    while t <= 1:
        while s <= 1:
            xt = (1-t)*x1 + t*x2
            yt = (1-t)*y1 + t*y2
            xs = (1-s)*x1 + s*x4
            ys = (1-s)*y1 + s*y4
            xa = (1-a)*x2 + a*x3
            ya = (1-a)*y2 + a*y3
            xb = (1-b)*x4 + b*x3
            yb = (1-b)*y4 + b*y3
            xi = (xt*(yt-yb)*(xs-xa)-xs*(ys-ya)*(xt-xb)+(ys-yt)*(xt-xb)*(xs-xa))/((yt-yb)*(xs-xa)-(ys-ya)*(xt-xb))
            yi = (ys*(xs-xa)*(yt-yb)-yt*(xt-xb)*(ys-ya)+(xt-xs)*(ys-ya)*(yt-yb))/((xs-xa)*(yt-yb)-(xt-xb)*(ys-ya))
            inter_pts.append((int(xi),int(yi)))
            s += res
            a += res
        s, a = 0, 0
        t += res
        b += res
    return inter_pts

# generate a path up to points given by path_limit starting from point given by object_loc
def generate_path(object_loc, panop_img, original_img, calib, rot_y, object_name, base_pts, got_path,
                  max_path_limit=200, min_path_limit=50, radius=0.5, res=0.175, surface_conf=1.0):
    drivable_pts = []
    drawable_pts = []
    curr_orient = rot_y % (2*np.pi)
    height, width, channels = original_img.shape
    # fake_img = np.copy(original_img)
    curr_x, y, curr_z, h = object_loc
    num_of_pts = 0
    all_steps = np.arange(0.1, radius+0.1, 0.1)

    # Pedestrains can go on both road and pavement
    # Cyclists and Cars can only go on road
    check_value = [50, 255] if object_name == 'Pedestrians' else [255]

    # a single 180 degree sweep in front has the following angles
    left_angles = [(curr_orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res))]
    right_angles = [(curr_orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res))]
    all_angles = left_angles + right_angles
    # print_angles = [degrees(i) for i in all_angles]
    # print(print_angles)
    # print([object_loc[0], object_loc[2]])
    new_base_pts = []
    for base_pt in base_pts:
        base_x, y, base_z, h = base_pt
        newx = base_x + radius * cos(curr_orient)
        newz = base_z + radius * sin(curr_orient)
        new_base_pts.append(np.array([newx, y, newz, h]))
    base_pts[:] = []
    base_pts = new_base_pts
    while num_of_pts < max_path_limit:
        num_of_pts_before = num_of_pts
        got_pt = False
        for angle in all_angles:
            # generate points that are offset from your current position
            # print(degrees(angle))
            for step in all_steps:
                candx = curr_x + step * cos(angle)
                candz = curr_z + step * sin(angle)
                point = np.dot(calib['P2'], np.array([candx, y, candz, h]))
                point = point[:2] / point[2]
                point = point.astype(np.int16)

                # check if point is within bounds of image plane when projected on it
                if (int(point[0]) < 0 or int(point[0]) >= width) or \
                        (int(point[1]) < 0 or int(point[1]) >= height):
                    continue

                # generate intermediate points within the base bounding box
                # check if all of them are on the desired drivable area
                bbox_pts = []
                for pt in base_pts:
                    base_x, y, base_z, h = pt
                    newx = base_x + step * cos(angle)
                    newz = base_z + step * sin(angle)
                    base_point = np.dot(calib['P2'], np.array([newx, y, newz, h]))
                    base_point = base_point[:2] / base_point[2]
                    base_point = base_point.astype(np.int16)
                    bbox_pts.append(base_point)
                if bbox_pts[0][0] == bbox_pts[1][0] == bbox_pts[2][0] == bbox_pts[3][0]:
                    bbox_pts[0][0] += 1.1
                    bbox_pts[1][0] += 2.1
                    bbox_pts[2][0] += 3.1
                    bbox_pts[3][0] += 4.1
                if bbox_pts[0][1] == bbox_pts[1][1] == bbox_pts[2][1] == bbox_pts[3][1]:
                    bbox_pts[0][1] += 1.1
                    bbox_pts[1][1] += 2.1
                    bbox_pts[2][1] += 3.1
                    bbox_pts[3][1] += 4.1
                inter_pts = generate_pts_in_rotated_rect(bbox_pts)
                total_pts = 0
                good_pts = 0
                fake_img = np.copy(original_img)
                for inter_pt in inter_pts:
                    pt2d = (int(inter_pt[0]), int(inter_pt[1]))
                    total_pts += 1
                    if (pt2d[0] < 0 or pt2d[0] >= width) or (pt2d[1] < 0 or pt2d[1] >= height):
                        continue
                    cv2.circle(fake_img, pt2d, 4, (255, 0, 255), -1)
                    if panop_img[pt2d[1], pt2d[0]] in check_value: good_pts += 1
                # cv2.imshow('FAKE', fake_img)
                # cv2.waitKey(1000)
                if good_pts / total_pts < surface_conf:
                    # print('UNSUCCESSFUL')
                    continue

                # update position of front point and base points
                drivable_pts.append(np.array([candx, y, candz, h]))
                curr_x, curr_z = candx, candz
                drawable_pts.append((point[0], point[1]))
                new_base_pts = []
                for base_pt in base_pts:
                    base_x, y, base_z, h = base_pt
                    newx = base_x + step * cos(angle)
                    newz = base_z + step * sin(angle)
                    new_base_pts.append(np.array([newx, y, newz, h]))
                base_pts[:] = []
                base_pts = new_base_pts
                # print([candx, candz])
                # print("SUCCESSFUL!")
                # print(len(drawable_pts))
                # cv2.circle(fake_img, (pt2dint[0], pt2dint[1]), 4, (255, 0, 255), -1)
                # cv2.imshow('FAKE', fake_img)
                # cv2.waitKey(100)
                num_of_pts += 1

                # generate new 180 degree sweep angles for the next position
                left_angles = [(angle+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res))]
                right_angles = [(angle-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res))]
                all_angles = left_angles + right_angles
                got_pt = True
                break
                # else:
                #     print([candx, candz])
                #     print("UNSUCCESSFUL!")
            all_steps = np.arange(0.1, 0.5 + 0.1, 0.1)
            if got_pt: break

        # means no drivable area was found in the full 180 degree sweep
        if num_of_pts == num_of_pts_before: break

    # only draw paths that have as many points as the lower limit
    if len(drawable_pts) >= min_path_limit:
        got_path = True
        for pt in drawable_pts: cv2.circle(original_img, pt, 4, (255, 0, 255), -1)
    return original_img, drivable_pts, got_path

def semantic_masks(seg_ids, sinfo, seg):
    for sid in seg_ids:
        _sinfo = sinfo.get(sid)
        if _sinfo is None or _sinfo["isthing"]:
            # Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
            continue
        yield (seg == sid).cpu().numpy().astype(np.bool), _sinfo

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_results(result):
    bbox_result, segm_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    masks = []
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > 0.3)[0]
        for i in inds:
            i = int(i)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            masks.append(mask)

    scores = bboxes[:, -1]
    inds = scores > 0.3
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    names = []
    for label_index in labels:
        label_text = coco_labels[label_index]
        names.append(label_text)

    return names, bboxes, masks

def get_2d_pt_from_vel(velo_pt, calib):
    P2 = calib['P2']
    R0 = np.eye(4)
    R0[:-1, :-1] = calib['R0_rect']
    Tr = np.eye(4)
    Tr[:-1, :] = calib['Tr_velo_to_cam']

    vld = velo_pt.T.reshape(4, 1)
    pt3d = vld[:, vld[-1, :] > 0].copy()
    pt3d[-1, :] = 1
    pt3d_cam = R0 @ Tr @ pt3d
    mask = pt3d_cam[2, :] >= 0  # Z >= 0
    pt2d_cam = P2 @ pt3d_cam[:, mask]
    pt2d = (pt2d_cam / pt2d_cam[2, :])[:-1, :].T
    return pt2d

def plot_3d_bbox(img, calib, bbox3d_center, bbox3d_dims, bbox3d_roty):
    box_3d = []
    box_pts = []
    P2 = calib['P2']

    for i in [1, -1]:
        for j in [1, -1]:
            for k in [0, 1]:
                point = np.copy(bbox3d_center)
                point[0] = bbox3d_center[0] + i * bbox3d_dims[1] / 2 * \
                           np.cos(-bbox3d_roty + np.pi / 2) + (j * i) * bbox3d_dims[2] / 2 * \
                           np.cos(-bbox3d_roty)
                point[2] = bbox3d_center[2] + i * bbox3d_dims[1] / 2 * \
                           np.sin(-bbox3d_roty + np.pi / 2) + (j * i) * bbox3d_dims[2] / 2 * \
                           np.sin(-bbox3d_roty)
                point[1] = bbox3d_center[1] - k * bbox3d_dims[0]

                point = np.append(point, 1)
                box_pts.append(point)
                point = np.dot(P2, point)
                point = point[:2] / point[2]
                point = point.astype(np.int16)
                box_3d.append(point)

    for i in range(4):
        point_1_ = box_3d[2 * i]
        point_2_ = box_3d[2 * i + 1]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,0,255), 1)

    for i in range(8):
        point_1_ = box_3d[i]
        point_2_ = box_3d[(i + 2) % 8]
        cv2.line(img, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0,0,255), 1)

    return img, box_pts

def calculate_iou(box_1, box_2):
    poly_1 = Polygon([[box_1[0], box_1[1]], [box_1[2], box_1[1]], [box_1[2], box_1[3]], [box_1[0], box_1[3]]])
    poly_2 = Polygon([[box_2[0], box_2[1]], [box_2[2], box_2[1]], [box_2[2], box_2[3]], [box_2[0], box_2[3]]])
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

cityscapes_labels = ['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
                     'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
                     'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river',
                     'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel',
                     'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind',
                     'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor',
                     'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock',
                     'wall', 'rug']

coco_labels = [
        'Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc',
                    default='/mnt/sdb1/datasets/kitti_tracking/training/0006', type=str,
                    help='your data folder location')
parser.add_argument('--velo_type', default='velodyne', type=str, help='choices: velodyne, fake_velodyne')
parser.add_argument('--operation_type', default='TRAJECTORY PRED', type=str,
                    help='choices: GATHER DATA, TRAJECTORY PRED, SANITY')
#for panoptic segmentation
parser.add_argument("--config-file",
                    default="detectron_2/configs/Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml",
                    metavar="FILE", help="path to config file")
parser.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="Minimum score for instance predictions to be shown")
parser.add_argument("--opts",
                    help="Modify config options using the command-line 'KEY VALUE' pairs",
                    default=['MODEL.WEIGHTS', 'detectron_2/model_final_be35db.pkl'],
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

#folder locations
left_img_folder = os.path.join(args.data_loc, 'image_2')
right_img_folder = os.path.join(args.data_loc, 'image_3')
calib_folder = os.path.join(args.data_loc, 'calib')
velo_folder = os.path.join(args.data_loc, args.velo_type)
label_folder = os.path.join(args.data_loc, 'label_2')

#create folders where training data and ground truth for 3d object detector will be stored
img_data_folder = os.path.join(args.data_loc, 'IMAGES')
velo_data_folder = os.path.join(args.data_loc, 'VELO')
gt_folder = os.path.join(args.data_loc, 'gt_labels')
pred_folder = os.path.join(args.data_loc, 'PREDS')
Path(pred_folder).mkdir(parents=True, exist_ok=True)
Path(gt_folder).mkdir(parents=True, exist_ok=True)
Path(img_data_folder).mkdir(parents=True, exist_ok=True)
Path(velo_data_folder).mkdir(parents=True, exist_ok=True)

#sort label, calib and velo files for parsing later on
label_files = natsort.natsorted(os.listdir(label_folder))
calib_files = natsort.natsorted(os.listdir(calib_folder))
velo_files = natsort.natsorted(os.listdir(velo_folder))

# label must follow KITTI Object Detection format
# [name truncated occluded alpha
# 2dbbox_x1 2d_bbox_y1 2d_bbox_x2 2d_bbox_y2
# 3d_bbox_height 3d_bbox_width 3d_bbox_length 3d_bbox_x 3d_bbox_y 3d_bbox_z rot_y]
frame_dict = {}
ignore_frame = []
for label_file in label_files[41:]:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
    if os.stat(os.path.join(label_folder, label_file)).st_size == 0:
        ignore_frame.append(label_file.split('.')[0])
        continue
    data = pd.read_csv(os.path.join(label_folder, label_file), sep=" ", header=None)
    data = list(data.to_numpy())
    all_valid_dets = []
    for label in data:
        if label[0] in ['Car', 'Pedestrian', 'Cyclist']:
            all_valid_dets.append(label)
    frame_num = label_file.split('.')[0]
    frame_dict[frame_num] = all_valid_dets

if args.operation_type in ['GATHER DATA', 'TRAJECTORY PRED']:
    #load model for instance segmentation
    config_file_inst = 'DetectoRS/configs/DetectoRS/DetectoRS_mstrain_400_1200_x101_32x4d_40e.py'
    checkpoint_file_inst = 'DetectoRS/DetectoRS_X101-ed983634.pth'
    model_inst = init_detector(config_file_inst, checkpoint_file_inst, device='cuda:0')

    #load model for panoptic segmentation
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    left_imgs = natsort.natsorted(os.listdir(left_img_folder))
    obj_counter = 0
    for left_img in left_imgs[41:]:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
        frame_num_str = left_img.split('.')[0]
        frame_num_int = int(frame_num_str)
        if frame_num_str in ignore_frame: continue

        # get calibration parameters for the frame
        clb_file = open(os.path.join(calib_folder, calib_files[frame_num_int]))
        clb = {}
        for line in clb_file:
            calib_line = line.split(':')
            if len(calib_line) < 2:
                continue
            key = calib_line[0]
            value = np.array(list(map(float, calib_line[1].split())))
            value = value.reshape((3, -1))
            clb[key] = value

        # load velodyne points
        velo_pts = np.fromfile(os.path.join(velo_folder, velo_files[frame_num_int]),
                               dtype=np.float32, count=-1).reshape([-1, 4])
        velo_pts = list(velo_pts)

        # load image
        loaded_img = cv2.imread(os.path.join(left_img_folder, left_img))
        image_2d = np.copy(loaded_img)
        height, width, channels = loaded_img.shape

        # instance segmentation
        result = inference_detector(model_inst, loaded_img)
        names, boxes, masks = get_results(result)

        all_dets = []
        for det_name, det_bbox, det_mask in zip(names, boxes, masks):
            # 3d object detector only trained to detect Car, Pedestrian and Cyclist
            if det_name not in ['Car', 'Pedestrian', 'Cyclist']:
                continue
            det_bbox = [int(i) for i in det_bbox]

            # find the bounding box in kitti label file that is closest to detected bounding box
            IoU_so_far = 0
            gt_label = None
            for label in frame_dict[frame_num_str]:
                label_bbox = [int(label[4]), int(label[5]), int(label[6]), int(label[7])]
                if label[0] != det_name: continue        # label must have same name as detected object
                calculated_iou = calculate_iou(det_bbox, label_bbox)
                if calculated_iou > 0.2 and calculated_iou > IoU_so_far:
                    IoU_so_far = calculated_iou
                    if gt_label is None:
                        gt_label = list(label)
                    else:
                        gt_label[:] = []
                        gt_label = list(label)

            # if detected object has no correspondence with any of the labels, then ignore it
            if gt_label is None: continue

            # display instance segmentation of object
            seg_img = np.zeros((height, width), np.uint8)
            seg_img[det_mask] = 255

            # project every 3d point onto image and check if it is close to pixel of current instance segmentation
            index = 0
            indices_to_ignore = []
            object_velo = []
            for v in velo_pts:
                pt2d = get_2d_pt_from_vel(v, clb)
                if pt2d.size != 0 and 0 <= int(pt2d[0][0]) < width and 0 <= int(pt2d[0][1]) < height:
                    pt2dint = (int(pt2d[0][0]), int(pt2d[0][1]))
                    if seg_img[pt2dint[1], pt2dint[0]] == 255:
                        indices_to_ignore.append(index)
                        object_velo.append(v)
                else:
                    indices_to_ignore.append(index)
                index += 1

            # remove velo points that have been assigned to current detected object
            for i in sorted(indices_to_ignore, reverse=True):
                del velo_pts[i]

            # if no point cloud for current object is found, then ignore it
            if not object_velo: continue

            # detected 2d and 3d bounding box and name
            gt_2d_bbox = gt_label[4:8]
            gt_3d_bbox = gt_label[8:14]
            gt_dist = np.linalg.norm(gt_label[11:14])
            gt_alpha = gt_label[3]
            gt_roty = gt_label[-1]
            gt_name = gt_label[0]
            all_dets.append({'name': gt_name, 'bbox2d': gt_2d_bbox, 'bbox3d': gt_3d_bbox,
                             'dist': gt_dist, 'alpha': gt_alpha, 'roty': gt_roty})

            if args.operation_type == 'GATHER DATA':
                # store velo data for training 3d object detector
                object_velo = np.array(object_velo)
                np.save(os.path.join(velo_data_folder, str(obj_counter) + '.npy'), object_velo)

                # crop object out of image and store it for training 3d object detector
                crop_img = image_2d[int(gt_2d_bbox[1]):int(gt_2d_bbox[3]), int(gt_2d_bbox[0]):int(gt_2d_bbox[2])]
                cv2.imwrite(os.path.join(img_data_folder, str(obj_counter) + '.png'), crop_img)

                # store gt_labels for training 3d object detector
                # gt_labels stored in same format as kitti
                # [name truncated occluded alpha
                # 2dbbox_x1 2d_bbox_y1 2d_bbox_x2 2d_bbox_y2
                # 3d_bbox_height 3d_bbox_width 3d_bbox_length 3d_bbox_x 3d_bbox_y 3d_bbox_z rot_y frame_num]
                # frame_num is just for sanity purposes
                gt_file = open(os.path.join(gt_folder, str(obj_counter) + '.txt'), "w+")
                gt_file.write(gt_name + ' ' + '-1' + ' ' + '-1' + ' ' + str(gt_alpha) + ' ')
                gt_file.write(str(gt_2d_bbox[0]) + ' ' + str(gt_2d_bbox[1]) + ' ' +
                              str(gt_2d_bbox[2]) + ' ' + str(gt_2d_bbox[3]) + ' ')
                gt_file.write(str(gt_3d_bbox[0]) + ' ' + str(gt_3d_bbox[1]) + ' ' + str(gt_3d_bbox[2]) + ' ' +
                              str(gt_3d_bbox[3]) + ' ' + str(gt_3d_bbox[4]) + ' ' + str(gt_3d_bbox[5]) + ' ' +
                              str(gt_roty) + ' ' + frame_num_str)
                gt_file.write('\n')
                gt_file.close()
                obj_counter += 1

        all_dets = sorted(all_dets, key=lambda k: k['dist'])
        if args.operation_type == 'TRAJECTORY PRED' and all_dets:
            # get panoptic predictions
            predictions, visualized_output = demo.run_on_image(loaded_img)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            sinfo = {s["id"]: s for s in segments_info}
            segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
            areas = areas.cpu().numpy()
            sorted_idxs = np.argsort(-areas)
            seg_ids, seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
            seg_ids = seg_ids.tolist()
            for sid, area in zip(seg_ids, seg_areas):
                if sid in sinfo:
                    sinfo[sid]["area"] = float(area)

            # only use road and pavement pixels
            display_tensor = torch.zeros_like(panoptic_seg)
            for mask, sinfo in semantic_masks(seg_ids, sinfo, panoptic_seg):
                category_idx = sinfo["category_id"]
                text = cityscapes_labels[category_idx]
                if text == 'road': display_tensor[mask] = 255
                if text == 'pavement': display_tensor[mask] = 50
            display_tensor = display_tensor.cpu().numpy().astype(np.uint8)

            # get drivable paths
            sufficient_path = False
            object_iter = 0
            while not sufficient_path:
                new_img = np.copy(loaded_img)
                # fake_img = np.copy(loaded_img)
                orient = abs(all_dets[object_iter]['roty'])
                res = 0.5
                left_angles = [(orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res)+1)]
                right_angles = [(orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res)+1)]
                all_angles = left_angles + right_angles
                # angle_degrees = [degrees(i) for i in all_angles]
                # print(angle_degrees)
                gt_bbox3d_dims = all_dets[object_iter]['bbox3d'][:3]
                gt_bbox3d_loc = all_dets[object_iter]['bbox3d'][3:]
                new_img, pts = plot_3d_bbox(new_img, clb, gt_bbox3d_loc, gt_bbox3d_dims, all_dets[object_iter]['roty'])
                pts = pts[0::2]
                front_pts = [pts[0], pts[3]]
                object_loc = sum(front_pts) / len(front_pts)
                object_loc_img = np.dot(clb['P2'], object_loc)
                object_loc_img = object_loc_img[:2] / object_loc_img[2]
                object_loc_img = object_loc_img.astype(np.int16)
                cv2.circle(new_img, (object_loc_img[0], object_loc_img[1]), 4, (255,0,255), -1)
                for angle in all_angles:
                    new_img, drivable_pts, sufficient_path = \
                        generate_path(object_loc, display_tensor, new_img, clb, angle,
                                      all_dets[object_iter]['name'], copy.deepcopy(pts),
                                      sufficient_path, radius=gt_bbox3d_dims[2] * 2)
                if object_iter < len(all_dets) - 1: object_iter += 1
                else: break

            # cd into folder containing images with trajectory predictions drawn on them
            # type following in console:
            # mencoder mf://*.png -mf fps=10:type=png -ovc x264 -x264encopts bitrate=1200:threads=2 -o outputfile.mkv
            cv2.imshow('BOTTOM', new_img)
            cv2.imwrite(os.path.join(args.data_loc, 'PREDS', left_img), new_img)
            cv2.imshow('PANOP', visualized_output.get_image()[:, :, ::-1])
            cv2.imshow('BLACK', display_tensor)
            cv2.waitKey(1)

        print("PROCESSED IMAGE " + left_img)

if args.operation_type == 'SANITY':
    original_imgs_folder = os.path.join(args.data_loc, 'image_2')
    calib_files_folder = os.path.join(args.data_loc, 'calib')
    image_folder = os.path.join(args.data_loc, 'IMAGES')
    velo_folder = os.path.join(args.data_loc, 'VELO')
    gt_folder = os.path.join(args.data_loc, 'gt_labels')

    original_imgs = natsort.natsorted(os.listdir(original_imgs_folder))
    calib_files = natsort.natsorted(os.listdir(calib_files_folder))
    images = natsort.natsorted(os.listdir(image_folder))
    velo_files = natsort.natsorted(os.listdir(velo_folder))
    gt_files = natsort.natsorted(os.listdir(gt_folder))

    # gt_labels stored in same format as kitti
    # [name truncated occluded alpha
    # 2dbbox_x1 2d_bbox_y1 2d_bbox_x2 2d_bbox_y2
    # 3d_bbox_height 3d_bbox_width 3d_bbox_length 3d_bbox_x 3d_bbox_y 3d_bbox_z rot_y frame_num]
    # frame_num is just for sanity purposes
    for gt_file, image, velo_file in zip(gt_files, images, velo_files):
        gt = pd.read_csv(os.path.join(gt_folder, gt_file), sep=" ", header=None)
        gt = list(gt.to_numpy())[0]

        gt_name = gt[0]
        gt_bbox2d = gt[4:8]
        gt_bbox3d_dims = gt[8:11]
        gt_bbox3d_loc = gt[11:14]
        gt_roty = gt[-2]
        frame = gt[-1]

        velo_coords = np.load(os.path.join(velo_folder, velo_file))
        crop_img = cv2.imread(os.path.join(image_folder, image))
        original_img2d = cv2.imread(os.path.join(original_imgs_folder, original_imgs[frame]))
        original_img3d = cv2.imread(os.path.join(original_imgs_folder, original_imgs[frame]))
        height, width, channels = original_img2d.shape

        partial_pt_cloud_img = np.zeros((height, width), np.uint8)  # display point cloud of object only

        clb_file = open(os.path.join(calib_files_folder, calib_files[frame]))
        clb = {}
        for line in clb_file:
            calib_line = line.split(':')
            if len(calib_line) < 2:
                continue
            key = calib_line[0]
            value = np.array(list(map(float, calib_line[1].split())))
            value = value.reshape((3, -1))
            clb[key] = value

        for v in velo_coords:
            pt2d = get_2d_pt_from_vel(v, clb)
            pt2dint = (int(pt2d[0][0]), int(pt2d[0][1]))
            cv2.circle(partial_pt_cloud_img, (pt2dint[0], pt2dint[1]), 1, 255, -1)

        pt1 = (int(gt_bbox2d[0]), int(gt_bbox2d[1]))
        pt2 = (int(gt_bbox2d[2]), int(gt_bbox2d[3]))
        cv2.rectangle(original_img2d, pt1, pt2, (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_img2d, gt_name, (pt1[0], pt1[1]-20), font, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('2D BBOX', original_img2d)
        original_img3d, _ = \
            plot_3d_bbox(original_img3d, clb, gt_bbox3d_loc, gt_bbox3d_dims, gt_roty)
        cv2.imshow('3D BBOX', original_img3d)
        cv2.imshow('PARTIAL POINT CLOUD', partial_pt_cloud_img)
        cv2.imshow('IMAGE CROP', crop_img)
        cv2.waitKey(5000)
