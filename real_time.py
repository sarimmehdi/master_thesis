import argparse
import cv2
import os
import pandas as pd
import natsort
import numpy as np
from colour import Color
from pathlib import Path
from math import sin, cos, atan2, degrees
from multiprocessing import Process

def get_pts_from_vel(velo_pts, calib):
    P2 = calib['P2']
    R0 = np.eye(4)
    R0[:-1, :-1] = calib['R0_rect']
    Tr = np.eye(4)
    Tr[:-1, :] = calib['Tr_velo_to_cam']

    vld = velo_pts.T.reshape(4, -1)
    road_ps = [is_road(vld[:,i], panop_img, P2, R0, Tr) for i in range(vld.shape[1])]
    road_pts = [{'pt': i[0], 'angle': round(atan2(i[0][0], i[0][2]), 2),
                 'dist': np.linalg.norm(i[0]), 'pixel_loc': i[1]} for i in road_ps if i[1] is not None]
    return road_pts

# quick function to check if velo point is a road point or not
def is_road(point, panop_img, P2, R0, Tr):
    height, width = panop_img.shape
    p = point.reshape(4,1)
    pt3d = p[:, p[-1, :] > 0].copy()
    pt3d[-1, :] = 1
    pt3d_cam = R0 @ Tr @ pt3d
    mask = pt3d_cam[2, :] >= 0  # Z >= 0
    pt2d_cam = P2 @ pt3d_cam[:, mask]
    pt2d = (pt2d_cam / pt2d_cam[2, :])[:-1, :].T.astype(int)
    pt2d = pt2d.flatten()
    if pt2d.size != 0:
        if 0 <= pt2d[0] < width and 0 <= pt2d[1] < height:
            if panop_img[pt2d[1], pt2d[0]] == 255: return (point, np.array([pt2d[0], pt2d[1]]))
    return (point, None)

# generate different perspective for non-velo points
def diff_persp_not_velo(pts, calib, offsetx, offsety, offsetz, invert_axes):
    points = pts.reshape((-1, 4))
    points[:,0], points[:,1], points[:,2] = points[:,0] + offsetx, \
                                            points[:,1] + offsety, \
                                            points[:,2] + offsetz
    axis1, axis2 = invert_axes
    points[:,axis1] = points[:,axis1]*cos(np.pi/18) - points[:,axis2]*sin(np.pi/18)
    points[:,axis2] = points[:,axis1]*sin(np.pi/18) + points[:,axis2]*cos(np.pi/18)

    pts2d_cam = calib['P2'] @ points.T
    pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T.astype(int)
    return pts2d

# generate different perspective for velo points
def diff_persp_is_velo(pts, panop_img, calib, offsetx, offsety, offsetz, invert_axes):
    points = pts.reshape((-1, 4))
    points[:,0], points[:,1], points[:,2] = points[:,0] + offsetx, \
                                            points[:,1] + offsety, \
                                            points[:,2] + offsetz
    axis1, axis2 = invert_axes
    points[:,axis1] = points[:,axis1]*cos(np.pi/18) - points[:,axis2]*sin(np.pi/18)
    points[:,axis2] = points[:,axis1]*sin(np.pi/18) + points[:,axis2]*cos(np.pi/18)

    P2 = calib['P2']
    R0 = np.eye(4)
    R0[:-1, :-1] = calib['R0_rect']
    Tr = np.eye(4)
    Tr[:-1, :] = calib['Tr_velo_to_cam']
    height, width = panop_img.shape

    points = points.T
    pts3d = points[:, points[-1, :] > 0].copy()
    pts3d[-1, :] = 1
    pts3d_cam = R0 @ Tr @ pts3d
    mask = pts3d_cam[2, :] >= 0  # Z >= 0
    pts2d_cam = P2 @ pts3d_cam[:, mask]
    pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T.astype(int)
    pts2d = pts2d[np.logical_and(np.logical_and(pts2d[:,0]>=0, pts2d[:,0]<width),
                                 np.logical_and(pts2d[:,1]>=0, pts2d[:,1]<height))]
    return pts2d

def get_img_pts(points, calib):
    pts2d_cam = calib['P2'] @ points.T
    pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T.astype(int)
    return pts2d

# bbox3d_roty must be between -pi and pi as is the case in the KITTI label format
def plot_3d_bbox(img, calib, bbox3d_center, bbox3d_dims, bbox3d_roty):
    h, w, l = bbox3d_dims
    x, y, z = bbox3d_center
    p0, p1, p2, p3 = np.array([l/2,0,w/2,1]), np.array([-l/2,0,w/2,1]), \
                     np.array([-l/2,0,-w/2,1]), np.array([l/2,0,-w/2,1])
    p4, p5, p6, p7 = np.array([l/2,-h,w/2,1]), np.array([-l/2,-h,w/2,1]), \
                     np.array([-l/2,-h,-w/2,1]), np.array([l/2,-h,-w/2,1])
    pts_array = np.array([p0, p1, p2, p3, p4, p5, p6, p7]).transpose()
    rot_mat = np.array([[cos(bbox3d_roty), 0, sin(bbox3d_roty), 0],[0,1,0,0],
                        [-sin(bbox3d_roty), 0, cos(bbox3d_roty), 0],[0,0,0,1]])
    pts_array = np.matmul(rot_mat, pts_array).transpose()+np.array([x,y,z,0])
    box_3d = get_img_pts(pts_array, calib)
    for i in range(4):
        pt1, pt2, pt3, pt4 = box_3d[i%4], box_3d[(i+1)%4], box_3d[(i+4)%8], box_3d[(i+5)%8]
        pt5, pt6 = box_3d[(i%4)+4], box_3d[((i+1)%4)+4]
        cv2.line(img, tuple(pt1), tuple(pt2), (0, 0, 255), 3)
        cv2.line(img, tuple(pt1), tuple(pt3), (0, 0, 255), 3)
        cv2.line(img, tuple(pt2), tuple(pt4), (0, 0, 255), 3)
        cv2.line(img, tuple(pt5), tuple(pt6), (0, 0, 255), 3)
    cv2.line(img, tuple(box_3d[0]), tuple(box_3d[-1]), (0, 0, 255), 3)
    cv2.line(img, tuple(box_3d[3]), tuple(box_3d[4]), (0, 0, 255), 3)
    center_pt_img = get_img_pts(np.append(bbox3d_center, 1).reshape(-1,4), calib)
    cv2.circle(img, tuple(center_pt_img[0]), 3, (255, 255, 255), -1)

    return pts_array

# generate evenly spaced points inside any rect
def generate_pts_in_rotated_rect(bbox_pts, res=0.3):
    x1, y1 = bbox_pts[0][0], bbox_pts[0][2]
    x2, y2 = bbox_pts[1][0], bbox_pts[1][2]
    x3, y3 = bbox_pts[2][0], bbox_pts[2][2]
    x4, y4 = bbox_pts[3][0], bbox_pts[3][2]

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
            den = (yt-yb)*(xs-xa)-(ys-ya)*(xt-xb)
            if den == 0: den = 1
            xi = (xt*(yt-yb)*(xs-xa)-xs*(ys-ya)*(xt-xb)+(ys-yt)*(xt-xb)*(xs-xa))/den
            den = (xs-xa)*(yt-yb)-(xt-xb)*(ys-ya)
            if den == 0: den = 1
            yi = (ys*(xs-xa)*(yt-yb)-yt*(xt-xb)*(ys-ya)+(xt-xs)*(ys-ya)*(yt-yb))/den
            inter_pts.append(np.array([xi,bbox_pts[0][1],yi,1]))
            s += res
            a += res
        s, a = 0, 0
        t += res
        b += res
    return inter_pts

def generate_position_heatmap(panop_img, original_img, pt_cld_img, temp_blank_image,
                              calib, rot_y, object_dict, offsets):
    # Pedestrians can go on road, pavement, and grass
    # Cyclists can go on road and pavement
    # Cars can only go on road
    if object_dict['name'] == 'Pedestrian': check_value = [50, 100, 255]
    if object_dict['name'] == 'Cyclist': check_value = [50, 255]
    if object_dict['name'] == 'Car': check_value = [255]

    angle = rot_y % (2 * np.pi)
    height, width, channels = original_img.shape
    bbox3d_dims, bbox3d_loc = object_dict['bbox3d'][:3], object_dict['bbox3d'][3:]
    first_dist = bbox3d_dims[2]
    start_steps = np.arange(0, first_dist, 0.1)
    mid_steps = np.arange(first_dist, first_dist + 1.0, 0.1)
    end_steps = np.arange(first_dist + 1.0, first_dist + 2.0, 0.1)
    step_types = [start_steps, mid_steps, end_steps]
    color_types = {'RED': list(Color("red").range_to(Color("yellow"),start_steps.shape[0])),
                   'YELLOW': list(Color("yellow").range_to(Color("orange"),mid_steps.shape[0])),
                   'BLUE': list(Color("orange").range_to(Color("blue"),end_steps.shape[0]))}
    status_types = ['DANGER_CLOSE', 'CAUTIOUS', 'SAFE']

    # rotate bounding box based on angle
    if 0 <= angle <= np.pi: theta = -angle
    if np.pi < angle <= 2*np.pi: theta = 2*np.pi - angle
    oriented_points = plot_3d_bbox(np.copy(original_img), calib, bbox3d_loc, bbox3d_dims, theta)
    bbox3d_loc = np.append(bbox3d_loc, 1)

    # project points forward and check if they lie on the road
    initial_step = True
    inter_pts = generate_pts_in_rotated_rect(oriented_points[:4])
    offsetx, offsety, offsetz = offsets
    for step_type, color_type, drivable_status in zip(step_types, color_types.values(), status_types):
        p_locs = [bbox3d_loc+np.array([step*cos(angle),0,step*sin(angle),0]) for step in step_type]
        b_locs = [inter_pt+np.array([step*cos(angle),0,step*sin(angle),0]) for step in step_type for
                  inter_pt in inter_pts]
        p_locs, b_locs = np.array(p_locs), np.array(b_locs)
        p_img_pts, b_img_pts = get_img_pts(p_locs, calib), get_img_pts(b_locs, calib)
        p_img_pts = p_img_pts[np.logical_and(np.logical_and(p_img_pts[:,0]>=0, p_img_pts[:,0]<width),
                                             np.logical_and(p_img_pts[:,1]>=0, p_img_pts[:,1]<height))]
        b_img_pts = b_img_pts[np.logical_and(np.logical_and(b_img_pts[:,0]>=0, b_img_pts[:,0]<width),
                                             np.logical_and(b_img_pts[:,1]>=0, b_img_pts[:,1]<height))]
        persp_pts = diff_persp_not_velo(p_locs, calib, offsetx, offsety, offsetz, [1, 2])
        if p_img_pts.size == 0 or b_img_pts.size == 0: continue
        iterator, skip_len = 0, b_img_pts.shape[0] // p_img_pts.shape[0]
        end_index = 0
        for p_pt, p_pt_persp in zip(p_img_pts, persp_pts):
            start_index = end_index
            end_index = start_index + iterator*skip_len
            rgb_val = color_type[iterator].rgb
            iterator += 1
            if panop_img[p_pt[1], p_pt[0]] in check_value and drivable_status == 'DANGER_CLOSE':
                cv2.circle(original_img, tuple(p_pt), 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                cv2.circle(pt_cld_img, tuple(p_pt), 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                cv2.circle(temp_blank_image, tuple(p_pt), 5, 50, -1)
                continue
            if panop_img[p_pt[1], p_pt[0]] not in check_value: continue
            if all(i in check_value for i in panop_img[b_img_pts[start_index:end_index,1], b_img_pts[start_index:end_index,0]]):
                cv2.circle(original_img, tuple(p_pt), 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                cv2.circle(pt_cld_img, tuple(p_pt), 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                if drivable_status == 'SAFE': cv2.circle(temp_blank_image, tuple(p_pt), 1, 255, -1)
                if drivable_status == 'CAUTIOUS': cv2.circle(temp_blank_image, tuple(p_pt), 5, 100, -1)
                if drivable_status == 'DANGER_CLOSE': cv2.circle(temp_blank_image, tuple(p_pt), 5, 50, -1)
            elif panop_img[b_img_pts[:,1], b_img_pts[:,0]].all() != 255 and not initial_step: break
        initial_step = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc',
                    default='/mnt/sdb1/datasets/kitti_tracking/training/0001', type=str,
                    help='your data folder location')
args = parser.parse_args()

#folder locations
left_img_folder = os.path.join(args.data_loc, 'image_2')
calib_folder = os.path.join(args.data_loc, 'calib')
velo_folder = os.path.join(args.data_loc, 'velodyne')
label_folder = os.path.join(args.data_loc, 'label_2')

allowable_dets = ['Car', 'Pedestrian', 'Cyclist']

road_class_folder = os.path.join(args.data_loc, 'ROAD_CLASSIFIER')
Path(road_class_folder).mkdir(parents=True, exist_ok=True)

# get panop images
panop_img_folder = os.path.join(args.data_loc, 'PANOP_IMGS')

# get detections
det_folder = os.path.join(args.data_loc, 'DETS3D')
det_files = natsort.natsorted(os.listdir(det_folder))

road_class_folder = os.path.join(args.data_loc, 'ROAD_CLASSIFIER')
Path(road_class_folder).mkdir(parents=True, exist_ok=True)

left_imgs = natsort.natsorted(os.listdir(left_img_folder))
for left_img in left_imgs:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
    frame_num_str = left_img.split('.')[0]
    frame_num_int = int(frame_num_str)

    # get calibration parameters for the frame
    clb_file = pd.read_csv(os.path.join(calib_folder, frame_num_str + '.txt'), sep=":", header=None)
    clb_file = list(clb_file.to_numpy())
    clb = {calib_line[0]: np.array(list(map(float, calib_line[1].split()))).reshape((3,-1))
           for calib_line in clb_file}

    # load image
    loaded_img = cv2.imread(os.path.join(left_img_folder, left_img))
    image_2d = np.copy(loaded_img)
    height, width, channels = loaded_img.shape

    # load detections
    if not os.path.exists(os.path.join(det_folder, frame_num_str+'.txt')) or \
            os.stat(os.path.join(det_folder, frame_num_str+'.txt')).st_size < 10:
        continue
    det = pd.read_csv(os.path.join(det_folder, frame_num_str+'.txt'), sep=" ", header=None)
    det = list(det.to_numpy())
    names = [label_line[0] for label_line in det if label_line[0] in allowable_dets]
    # last element is score
    boxes = [label_line[8:15] for label_line in det if label_line[0] in allowable_dets]
    if not boxes: continue

    # load image with panoptic segmentations on it and overlay road pixels on original image
    if not os.path.exists(os.path.join(panop_img_folder, left_img)): continue
    panop_img = cv2.imread(os.path.join(panop_img_folder, left_img))
    drivable_img = np.copy(panop_img)
    drivable_img[np.where((drivable_img == [255, 255, 255]).all(axis=2))] = [255, 0, 0]
    drivable_img[np.where((drivable_img == [50, 50, 50]).all(axis=2))] = [0, 0, 0]
    drivable_img[np.where((drivable_img == [100, 100, 100]).all(axis=2))] = [0, 0, 0]
    loaded_img = cv2.addWeighted(loaded_img, 1.0, drivable_img, 0.5, 0)
    panop_img = cv2.cvtColor(panop_img, cv2.COLOR_BGR2GRAY)

    # load velodyne points
    if not os.path.exists(os.path.join(velo_folder, frame_num_str+'.bin')): continue
    velo_pts = np.fromfile(os.path.join(velo_folder, frame_num_str+'.bin'),
                           dtype=np.float32, count=-1).reshape([-1, 4])

    # draw point cloud from a top down perspective
    pt_cloud_img = np.zeros((height, width, channels), np.uint8)
    pts2d_persp = diff_persp_is_velo(velo_pts, panop_img, clb, 5, 0, -2, [0,2])
    original_road_pts = get_pts_from_vel(velo_pts, clb)
    pt_cloud_img[pts2d_persp[:,1], pts2d_persp[:,0]] = (255,255,255)

    # store all 3d detections in a list of dictionaries
    all_dets = [{'name': name, 'roty': box[-1], 'bbox3d': box[:6], 'dist': np.linalg.norm(box[3:6])}
                for name, box in zip(names, boxes)]

    all_dets = sorted(all_dets, key=lambda k: k['dist'])
    if len(all_dets) >= 3: all_dets = all_dets[:3]
    object_iter = 0

    # draw heatmap
    temp_blank_image = np.zeros((height, width), np.uint8)
    while True:
        theta = all_dets[object_iter]['roty']
        name = all_dets[object_iter]['name']
        bbox3d_dims, bbox3d_loc = all_dets[object_iter]['bbox3d'][:3], all_dets[object_iter]['bbox3d'][3:]

        # object facing away from you has negative rotation (anti-clockwise from positive z axis)
        # object facing towards you has positive rotation (clockwise from positive z axis)
        orient = abs(theta) if theta <= 0 else 2*np.pi - theta
        res = np.pi/180
        left_angles = [(orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res)+1)]
        right_angles = [(orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res)+1)]
        all_angles = left_angles + right_angles

        # create bounding box in the beginning
        points = plot_3d_bbox(loaded_img, clb, bbox3d_loc, bbox3d_dims, all_dets[object_iter]['roty'])
        points2d = diff_persp_not_velo(points, clb, 0, 2, 5, [1, 2])
        for i in range(4):
            pt1, pt2, pt3, pt4 = points2d[i%4], points2d[(i+1)%4], points2d[(i+4)%8], points2d[(i+5)%8]
            pt5, pt6 = points2d[(i%4)+4], points2d[((i+1)%4)+4]
            cv2.line(pt_cloud_img, tuple(pt1), tuple(pt2), (0, 0, 255), 3)
            cv2.line(pt_cloud_img, tuple(pt1), tuple(pt3), (0, 0, 255), 3)
            cv2.line(pt_cloud_img, tuple(pt2), tuple(pt4), (0, 0, 255), 3)
            cv2.line(pt_cloud_img, tuple(pt5), tuple(pt6), (0, 0, 255), 3)
        cv2.line(pt_cloud_img, tuple(points2d[0]), tuple(points2d[-1]), (0, 0, 255), 3)
        cv2.line(pt_cloud_img, tuple(points2d[3]), tuple(points2d[4]), (0, 0, 255), 3)
        center2d = diff_persp_not_velo(np.append(bbox3d_loc, 1), clb, 0, 2, 5, [1, 2])
        cv2.circle(pt_cloud_img, tuple(center2d[0]), 3, (255, 255, 255), -1)
        processes = []
        for angle in all_angles:
            input_args = (panop_img, loaded_img, pt_cloud_img, temp_blank_image, clb, angle,
                          all_dets[object_iter], [0, 2, 5],)
            proc = Process(target=generate_position_heatmap, args=input_args)
            processes.append(proc)
            proc.start()
            # generate_position_heatmap(panop_img, loaded_img, pt_cloud_img, temp_blank_image,
            #                           clb, angle, all_dets[object_iter], [0, 2, 5])
        for p in processes: p.join()
        if object_iter < len(all_dets) - 1: object_iter += 1
        else: break     # if you have arrived at the end of your object list, then leave

    road_pts = [rp for rp in original_road_pts
                if temp_blank_image[rp['pixel_loc'][1], rp['pixel_loc'][0]] not in [100, 50] and rp['dist'] <= 20]
    for rp in road_pts: cv2.circle(pt_cloud_img, tuple(rp['pixel_loc']), 1, (255, 0, 0), -1)

    # another_temp_blank_image = np.zeros((height, width), np.uint8)
    # for r in road_pts: cv2.circle(another_temp_blank_image, tuple(r['pixel_loc']), 1, (255, 0, 0), -1)
    # cv2.imshow('1', another_temp_blank_image)
    # cv2.imshow('2', temp_blank_image)
    # cv2.waitKey(100000)

    if road_pts:
        road_pts = sorted(road_pts, key=lambda k: k['angle'])
        # give speed and steering recommendations here
        speed, steering = 'AS YOU WISH', np.pi/2
        if len(road_pts) / len(original_road_pts) <= 0.7: speed = 'MODERATE'
        if len(road_pts) / len(original_road_pts) <= 0.3: speed = 'SLOW DOWN!'
        if (speed == 'MODERATE' or speed == 'SLOW DOWN!') and len(road_pts) != 0:
            steering = sum(item['angle'] for item in road_pts) / len(road_pts)
        # if steering > np.pi/2:
        #     selected_pts = [r for r in road_pts if r['angle'] >= np.pi/2]
        #     if selected_pts: steering = selected_pts[0]['angle']
        # else:
        #     selected_pts = [r for r in road_pts if r['angle'] < np.pi/2]
        #     if selected_pts: steering = selected_pts[-1]['angle']
        print(degrees(steering))
        font = cv2.FONT_HERSHEY_SIMPLEX
        speed_text = 'RECOMMENDED SPEED: ' + speed
        steer_text = 'RECOMMENDED STEERING:'
        speed_text_loc = (2*width//3, 50)
        steer_text_loc = (2*width//5, 80)
        cv2.putText(loaded_img, speed_text, speed_text_loc, font, 0.6, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(loaded_img, speed_text, speed_text_loc, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(loaded_img, steer_text, steer_text_loc, font, 0.6, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(loaded_img, steer_text, steer_text_loc, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
        circle_center = (width//2, height//2)
        circle_radius = 100
        cv2.circle(loaded_img, circle_center, circle_radius, (0,0,0), 4)
        cv2.circle(loaded_img, circle_center, circle_radius, (255,255,255), 1)
        p = [int(circle_center[0] + circle_radius*cos(steering)),
             int(circle_center[1] + circle_radius*sin(steering))]
        y_rel = p[1] - circle_center[1]
        p[1] -= 2*y_rel
        cv2.line(loaded_img, circle_center, tuple(p), (0,0,0), 4)
        cv2.line(loaded_img, circle_center, tuple(p), (255,255,255), 1)

    # draw heatmap legend on image
    start_steps = np.arange(0, 1.0, 0.1)
    mid_steps = np.arange(1.0, 2.0, 0.1)
    end_steps = np.arange(2.0, 4.0, 0.1)
    step_types = [start_steps, mid_steps, end_steps]
    color_types = {'RED': list(Color("red").range_to(Color("yellow"),start_steps.shape[0])),
                   'YELLOW': list(Color("yellow").range_to(Color("orange"),mid_steps.shape[0])),
                   'BLUE': list(Color("orange").range_to(Color("blue"),end_steps.shape[0]))}
    text_names = ['DANGER-CLOSE', 'CAUTIOUS', 'SAFE']
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter = 0
    for all_steps, color_type, text in zip(step_types, color_types.values(), text_names):
        for step, color_val in zip(all_steps, color_type):
            rgb_val = color_val.rgb
            rgb_val = (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256))
            x_val = int(50 + (step/all_steps[-1])*50) + 10*counter
            cv2.circle(pt_cloud_img, (x_val,50), 10, rgb_val, -1)
            cv2.circle(loaded_img, (x_val, 50), 10, rgb_val, -1)

            if counter in [0, 15, 39]:
                x_val = 50 +10*counter
                color_ind = 0 if counter == 0 or counter == 15 else -1
                text_rgb = color_type[color_ind].rgb
                text_rgb = (int(text_rgb[2]*256), int(text_rgb[1]*256), int(text_rgb[0]*256))
                cv2.putText(loaded_img, text, (x_val, 80), font, 0.5, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(pt_cloud_img, text, (x_val, 80), font, 0.5, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(loaded_img, text, (x_val, 80), font, 0.5, text_rgb, 2, cv2.LINE_AA)
                cv2.putText(pt_cloud_img, text, (x_val, 80), font, 0.5, text_rgb, 2, cv2.LINE_AA)
            counter += 1
    cv2.imshow('BBOXES AND POINT CLOUD', cv2.vconcat([loaded_img, pt_cloud_img]))
    cv2.imwrite(os.path.join(road_class_folder, left_img), cv2.vconcat([loaded_img, pt_cloud_img]))
    print(left_img)
    cv2.waitKey(1)