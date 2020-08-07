import argparse
import os
import natsort
import cv2
import mmcv
import torch
import random
import copy
import pandas as pd
import numpy as np
import pycocotools.mask as maskUtils
from sort import *
from math import sin, cos, degrees
from pathlib import Path
from shapely.geometry import Polygon
from DetectoRS.mmdet.apis import init_detector, inference_detector
from colour import Color

# generate different perspective
def diff_persp(point, calib, offsetx, offsety, offsetz, invert_axes, is_velo=False):
    new_point = np.array([point[0]+offsetx, point[1]+offsety, point[2]+offsetz, point[3]])
    axis1, axis2 = invert_axes
    new_point[axis1] = new_point[axis1]*cos(np.pi*10/180) - new_point[axis2]*sin(np.pi*10/180)
    new_point[axis2] = new_point[axis1]*sin(np.pi*10/180) + new_point[axis2]*cos(np.pi*10/180)
    if not is_velo: return new_point

    P2 = calib['P2']
    R0 = np.eye(4)
    R0[:-1, :-1] = calib['R0_rect']
    Tr = np.eye(4)
    Tr[:-1, :] = calib['Tr_velo_to_cam']

    vld = new_point.T.reshape(4, 1)
    pt3d = vld[:, vld[-1, :] > 0].copy()
    pt3d[-1, :] = 1
    pt3d_cam = R0 @ Tr @ pt3d
    mask = pt3d_cam[2, :] >= 0  # Z >= 0
    pt2d_cam = P2 @ pt3d_cam[:, mask]
    pt2d = (pt2d_cam / pt2d_cam[2, :])[:-1, :].T
    return pt2d

# generate evenly spaced points inside any rect
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
            den = (yt-yb)*(xs-xa)-(ys-ya)*(xt-xb)
            if den == 0: den = 1
            xi = (xt*(yt-yb)*(xs-xa)-xs*(ys-ya)*(xt-xb)+(ys-yt)*(xt-xb)*(xs-xa))/den
            den = (xs-xa)*(yt-yb)-(xt-xb)*(ys-ya)
            if den == 0: den = 1
            yi = (ys*(xs-xa)*(yt-yb)-yt*(xt-xb)*(ys-ya)+(xt-xs)*(ys-ya)*(yt-yb))/den
            inter_pts.append((int(xi),int(yi)))
            s += res
            a += res
        s, a = 0, 0
        t += res
        b += res
    return inter_pts

def generate_position_heatmap(panop_img, original_img, blank_image, calib, rot_y, object_dict, surface_conf=1.0):
    # Pedestrians can go on road, pavement, and grass
    # Cyclists can go on road and pavement
    # Cars can only go on road
    if object_dict['name'] == 'Pedestrian': check_value = [50, 100, 255]
    if object_dict['name'] == 'Cyclist': check_value = [50, 255]
    if object_dict['name'] == 'Car': check_value = [255]

    angle = rot_y % (2 * np.pi)
    height, width, channels = original_img.shape
    bbox3d_dims, bbox3d_loc = object_dict['bbox3d'][:3], object_dict['bbox3d'][3:]
    curr_x, y, curr_z = bbox3d_loc
    first_dist = bbox3d_dims[2]
    start_steps = np.arange(0, first_dist, 0.1)
    mid_steps = np.arange(first_dist, first_dist + 1.0, 0.1)
    end_steps = np.arange(first_dist + 1.0, first_dist + 2.0, 0.1)
    step_types = [start_steps, mid_steps, end_steps]
    color_types = {'RED': list(Color("red").range_to(Color("yellow"),start_steps.shape[0])),
                   'YELLOW': list(Color("yellow").range_to(Color("orange"),mid_steps.shape[0])),
                   'BLUE': list(Color("orange").range_to(Color("blue"),end_steps.shape[0]))}

    # rotate bounding box based on angle
    if 0 <= angle <= np.pi: theta = -angle
    if np.pi < angle <= 2*np.pi: theta = 2*np.pi - angle
    oriented_points = plot_3d_bbox(np.copy(original_img), clb, bbox3d_loc, bbox3d_dims, theta)
    base_pts = oriented_points[:4]
    top_pts = oriented_points[4:]

    initial_step = True
    for all_steps, color_type in zip(step_types, color_types.values()):
        got_pt = False
        for step, color_val in zip(all_steps, color_type):
            candx = curr_x + step * cos(angle)
            candz = curr_z + step * sin(angle)
            if candz < 2: continue
            point = np.array([candx, y, candz, 1])
            point_normal = get_img_pt(point, calib)

            # check if point is within bounds of image plane when projected on it
            if (point_normal[0] < 0 or point_normal[0] >= width) or 
                    (point_normal[1] < 0 or point_normal[1] >= height):
                continue

            if initial_step and panop_img[point_normal[1], point_normal[0]] not in check_value: continue
            elif initial_step and panop_img[point_normal[1], point_normal[0]] in check_value:
                rgb_val = color_val.rgb
                cv2.circle(blank_image, point_normal, 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                got_pt = True
                continue

            # project bounding box in a direction
            base_bbox_pts = []
            top_bbox_pts = []
            for base_pt, top_pt in zip(base_pts, top_pts):
                base_x, base_y, base_z, base_h = base_pt
                newx = base_x + step * cos(angle)
                newz = base_z + step * sin(angle)
                base_bbox_pts.append(get_img_pt(np.array([newx, base_y, newz, base_h]), calib))
                top_x, top_y, top_z, top_h = top_pt
                newx = top_x + step * cos(angle)
                newz = top_z + step * sin(angle)
                top_bbox_pts.append(get_img_pt(np.array([newx, top_y, newz, top_h]), calib))

            inter_pts = generate_pts_in_rotated_rect(base_bbox_pts)
            total_pts = 0
            good_pts = 0
            for inter_pt in inter_pts:
                pt2d = (int(inter_pt[0]), int(inter_pt[1]))
                total_pts += 1
                if (pt2d[0] < 0 or pt2d[0] >= width) or (pt2d[1] < 0 or pt2d[1] >= height): continue
                if panop_img[pt2d[1], pt2d[0]] in check_value: good_pts += 1
            if good_pts / total_pts < surface_conf: continue

            if panop_img[point_normal[1], point_normal[0]] not in check_value: continue
            elif panop_img[point_normal[1], point_normal[0]] in check_value:
                rgb_val = color_val.rgb
                cv2.circle(blank_image, point_normal, 3, (int(rgb_val[2]*256), int(rgb_val[1]*256), int(rgb_val[0]*256)), -1)
                got_pt = True

        if not got_pt: break
        initial_step = False

def generate_potential_field(object_loc, panop_img, original_img, pt_cloud_img, calib, rot_y,
                             object_dict, radius, min_path_limit=10, surface_conf=0.5):
    offsets = [0, 2, 5]
    offsetx, offsety, offsetz = offsets

    drawable_pts = []
    drawable_pts_pt_cloud = []
    drivable_pts = []

    # Pedestrians can go on road, pavement, and grass
    # Cyclists can go on road and pavement
    # Cars can only go on road
    if object_dict['name'] == 'Pedestrian': check_value = [50, 100, 255]
    if object_dict['name'] == 'Cyclist': check_value = [50, 255]
    if object_dict['name'] == 'Car': check_value = [255]

    all_steps = np.arange(0.1, radius + 0.1, 0.1)
    angle = rot_y % (2 * np.pi)
    height, width, channels = original_img.shape
    curr_x, y, curr_z, h = object_loc
    bbox3d_dims, bbox3d_loc = object_dict['bbox3d'][:3], object_dict['bbox3d'][3:]

    if 0 <= angle <= np.pi: theta = -angle
    if np.pi < angle <= 2*np.pi: theta = 2*np.pi - angle
    oriented_points = plot_3d_bbox(np.copy(original_img), clb, bbox3d_loc, bbox3d_dims, theta)
    base_pts = oriented_points[:4]
    top_pts = oriented_points[4:]
    while True:
        got_pt = False
        for step in all_steps:
            candx = curr_x + step * cos(angle)
            candz = curr_z + step * sin(angle)
            if candz < 2: continue
            point = np.array([candx, y, candz, h])
            point_normal = get_img_pt(point, calib)
            point_pt_cld = get_img_pt(diff_persp(point, calib, offsetx, offsety, offsetz, [1, 2]), calib)

            # check if point is within bounds of image plane when projected on it
            if (point_normal[0] < 0 or point_normal[0] >= width) or 
                    (point_normal[1] < 0 or point_normal[1] >= height):
                continue

            # project bounding box in a direction
            base_bbox_pts = []
            top_bbox_pts = []
            new_base = []
            new_top = []
            for base_pt, top_pt in zip(base_pts, top_pts):
                base_x, base_y, base_z, base_h = base_pt
                newx = base_x + step * cos(angle)
                newz = base_z + step * sin(angle)
                new_base.append(np.array([newx, base_y, newz, base_h]))
                base_bbox_pts.append(get_img_pt(np.array([newx, base_y, newz, base_h]), calib))
                top_x, top_y, top_z, top_h = top_pt
                newx = top_x + step * cos(angle)
                newz = top_z + step * sin(angle)
                new_top.append(np.array([newx, top_y, newz, top_h]))
                top_bbox_pts.append(get_img_pt(np.array([newx, top_y, newz, top_h]), calib))

            inter_pts = generate_pts_in_rotated_rect(base_bbox_pts)
            total_pts = 0
            good_pts = 0
            for inter_pt in inter_pts:
                pt2d = (int(inter_pt[0]), int(inter_pt[1]))
                total_pts += 1
                if (pt2d[0] < 0 or pt2d[0] >= width) or (pt2d[1] < 0 or pt2d[1] >= height): continue
                if panop_img[pt2d[1], pt2d[0]] in check_value: good_pts += 1
            if good_pts / total_pts < surface_conf: continue

            if panop_img[point_normal[1], point_normal[0]] not in check_value: continue

            # update position of 3d bbox
            base_pts[:] = []
            base_pts = new_base
            top_pts[:] = []
            top_pts = new_top

            # update position
            drivable_pts.append(np.array([candx, y, candz, h]))
            curr_x, curr_z = candx, candz
            drawable_pts.append(point_normal)
            drawable_pts_pt_cloud.append(point_pt_cld)
            got_pt = True

        if not got_pt: break
        all_steps = np.arange(0.1, 0.2, 0.01)
        if len(drivable_pts) > 20: break

    # draw potential field
    for normal_pt, cld_pt in zip(drawable_pts, drawable_pts_pt_cloud):
        cv2.circle(original_img, normal_pt, 3, (0, 0, 255), -1)
        cv2.circle(pt_cloud_img, cld_pt, 3, (0, 0, 255), -1)
    got_path = True if len(drawable_pts) >= min_path_limit else False

    return got_path


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
    return pt2d, pt3d_cam[:, mask]

def get_img_pt(pt, calib):
    projected_point = np.dot(calib['P2'], pt)
    projected_point = projected_point[:2] / projected_point[2]
    projected_point = projected_point.astype(np.int16)
    return (projected_point[0], projected_point[1])

def plot_base_and_top(original_img, pt_cloud_img, base_pts, top_pts, calib, offsets):
    pt_cld_base_pts = []
    pt_cld_top_pts = []
    base_bbox_pts = []
    top_bbox_pts = []
    offsetx, offsety, offsetz = offsets

    # project points into image plane of original image and point cloud image
    for base_pt, top_pt in zip(base_pts, top_pts):
        base_bbox_pts.append(get_img_pt(base_pt, calib))
        top_bbox_pts.append(get_img_pt(top_pt, calib))
        b_pt = diff_persp(base_pt, calib, offsetx, offsety, offsetz, [1, 2])
        pt_cld_base_pts.append(get_img_pt(b_pt, calib))
        t_pt = diff_persp(top_pt, calib, offsetx, offsety, offsetz, [1, 2])
        pt_cld_top_pts.append(get_img_pt(t_pt, calib))

    for i in range(4):
        bp1, tp1 = base_bbox_pts[i%4], top_bbox_pts[i%4]
        bp2, tp2 = base_bbox_pts[(i+1)%4], top_bbox_pts[(i+1)%4]
        cv2.line(original_img, bp1, bp2, (0,0,255), 1)
        cv2.line(original_img, tp1, tp2, (0,0,255), 1)
        cv2.line(original_img, bp1, tp1, (0,0,255), 1)
        cv2.line(original_img, bp2, tp2, (0,0,255), 1)

        bp1, tp1 = pt_cld_base_pts[i%4], pt_cld_top_pts[i%4]
        bp2, tp2 = pt_cld_base_pts[(i+1)%4], pt_cld_top_pts[(i+1)%4]
        cv2.line(pt_cloud_img, bp1, bp2, (0,0,255), 3)
        cv2.line(pt_cloud_img, tp1, tp2, (0,0,255), 3)
        cv2.line(pt_cloud_img, bp1, tp1, (0,0,255), 3)
        cv2.line(pt_cloud_img, bp2, tp2, (0,0,255), 3)

    # indicate rotation of 3d bounding boxes
    b1, b2, t1, t2 = base_bbox_pts[0], base_bbox_pts[-1], top_bbox_pts[0], top_bbox_pts[-1]
    cv2.line(original_img, b1, t2, (0,0,255), 1)
    cv2.line(original_img, b2, t1, (0,0,255), 1)
    b1, b2, t1, t2 = pt_cld_base_pts[0], pt_cld_base_pts[-1], pt_cld_top_pts[0], pt_cld_top_pts[-1]
    cv2.line(pt_cloud_img, b1, t2, (0,0,255), 3)
    cv2.line(pt_cloud_img, b2, t1, (0,0,255), 3)

# generate a path up to points given by path_limit starting from point given by object_loc
def generate_path(panop_img, original_img, calib, rot_y, object_dict, got_path, radius, pt_cloud_img,
                  left_angle_limit, right_angle_limit,
                  max_path_limit=50, min_path_limit=50, res=0.5, surface_conf=1.0):
    curr_orient = rot_y % (2*np.pi)
    height, width, channels = original_img.shape
    num_of_pts = 0
    all_steps = np.arange(0.1, radius+0.1, 0.1)

    drivable_pts = []
    drawable_pts = []
    drawable_pts_pt_cloud = []
    offsets = [0,2,5]
    offsetx, offsety, offsetz = offsets

    # Pedestrians can go on road, pavement, and grass
    # Cyclists can go on road and pavement
    # Cars can only go on road
    if object_dict['name'] == 'Pedestrian': check_value = [50, 100, 255]
    if object_dict['name'] == 'Cyclist': check_value = [50, 255]
    if object_dict['name'] == 'Car': check_value = [255]

    # a single 180 degree sweep in front has the following angles
    t_left_angles = [(curr_orient+i*res) for i in range(int((np.pi/2) // res))
                     if (curr_orient+i*res) < left_angle_limit]
    t_right_angles = [(curr_orient-i*res) for i in range(1,int((np.pi/2) // res))
                      if (curr_orient-i*res) > right_angle_limit]
    left_angles = [i%(2*np.pi) for i in t_left_angles]
    right_angles = [i%(2*np.pi) for i in t_right_angles]
    all_angles = left_angles + right_angles

    # video_array = []
    # size = (width, height*3)
    # out = cv2.VideoWriter(os.path.join(os.getcwd(), 'how_it_works.avi'),
    #                       cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    # _original_img = np.copy(original_img)
    # _pt_cloud_img = np.copy(pt_cloud_img)
    bbox3d_dims, bbox3d_loc = object_dict['bbox3d'][:3], object_dict['bbox3d'][3:]
    points = plot_3d_bbox(np.copy(original_img), clb, bbox3d_loc, bbox3d_dims, object_dict['roty'])
    plot_base_and_top(original_img, pt_cloud_img, points[:4], points[4:], calib, offsets)
    base_pts = []
    top_pts = []
    # global_counter = 0
    while num_of_pts < max_path_limit:
        num_of_pts_before = num_of_pts
        got_pt = False
        for angle in all_angles:
            if 0 <= angle <= np.pi: theta = -angle
            if np.pi < angle <= 2*np.pi: theta = 2*np.pi - angle
            points = plot_3d_bbox(np.copy(original_img), clb, bbox3d_loc, bbox3d_dims, theta)
            front_pts = [points[0], points[3]]
            object_loc = sum(front_pts) / len(front_pts)
            curr_x, y, curr_z, h = object_loc
            drawable_pts.append(get_img_pt(object_loc, calib))
            drawable_pts_pt_cloud.append(get_img_pt(diff_persp(object_loc, calib, offsetx, offsety, offsetz, [1, 2]),
                                                    calib))
            base_pts = points[:4]
            top_pts = points[4:]

            # generate points that are offset from your current position
            for step in all_steps:
                candx = curr_x + step * cos(angle)
                candz = curr_z + step * sin(angle)
                if candz < 2: continue      # points behind camera
                point = np.array([candx, y, candz, h])
                point_normal = get_img_pt(point, calib)
                point_pt_cld = get_img_pt(diff_persp(point, calib, offsetx, offsety, offsetz, [1,2]), calib)

                # check if point is within bounds of image plane when projected on it
                if (int(point_normal[0]) < 0 or int(point_normal[0]) >= width) or 
                        (int(point_normal[1]) < 0 or int(point_normal[1]) >= height):
                    continue

                # project bounding box in a direction
                base_bbox_pts = []
                top_bbox_pts = []
                new_base = []
                new_top = []
                for base_pt, top_pt in zip(base_pts, top_pts):
                    base_x, base_y, base_z, base_h = base_pt
                    newx = base_x + step * cos(angle)
                    newz = base_z + step * sin(angle)
                    new_base.append(np.array([newx, base_y, newz, base_h]))
                    base_bbox_pts.append(get_img_pt(np.array([newx, base_y, newz, base_h]), calib))
                    top_x, top_y, top_z, top_h = top_pt
                    newx = top_x + step * cos(angle)
                    newz = top_z + step * sin(angle)
                    new_top.append(np.array([newx, top_y, newz, top_h]))
                    top_bbox_pts.append(get_img_pt(np.array([newx, top_y, newz, top_h]), calib))

                # check whether most/ all points of the base of new bounding box lie on the specified
                # drivable path or not
                # fake1 = np.copy(_original_img)
                # fake2 = np.copy(_pt_cloud_img)
                # fake3 = cv2.cvtColor(np.copy(panop_img), cv2.COLOR_GRAY2RGB)
                # plot_base_and_top(fake1, fake2, new_base, new_top, calib, offsets)
                # plot_base_and_top(fake3, fake2, new_base, new_top, calib, offsets)
                # cv2.circle(fake1, point_normal, 3, (255, 0, 255), -1)
                # cv2.circle(fake3, point_normal, 3, (255, 0, 255), -1)
                # temp_bbox3d_loc = sum(new_base)/len(new_base)
                # temp_bbox3d_loc = temp_bbox3d_loc[:-1]
                # center_pt_img = get_img_pt(np.append(temp_bbox3d_loc, 1), calib)
                # cv2.circle(fake1, center_pt_img, 3, (255, 255, 255), -1)
                # cv2.circle(fake3, center_pt_img, 3, (255, 255, 255), -1)

                inter_pts = generate_pts_in_rotated_rect(base_bbox_pts)
                total_pts = 0
                good_pts = 0
                for inter_pt in inter_pts:
                    pt2d = (int(inter_pt[0]), int(inter_pt[1]))
                    total_pts += 1
                    if (pt2d[0] < 0 or pt2d[0] >= width) or (pt2d[1] < 0 or pt2d[1] >= height): continue
                    if panop_img[pt2d[1], pt2d[0]] in check_value: good_pts += 1
                    # cv2.circle(fake1, pt2d, 2, (255, 0, 255), -1)
                    # cv2.circle(fake2, pt2d, 2, (255, 0, 255), -1)
                    # cv2.circle(fake3, pt2d, 2, (255, 0, 255), -1)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(fake1,'IoU with road: ' + str(good_pts / total_pts),
                #             (center_pt_img[0],center_pt_img[1]-20),font,1,(255,0,0),2,cv2.LINE_AA)
                # cv2.putText(fake3,'IoU with road: ' + str(good_pts / total_pts),
                #             (center_pt_img[0],center_pt_img[1]-20),font,1,(255,0,0),2,cv2.LINE_AA)
                # cv2.imshow('FAKE1', cv2.vconcat([fake1, fake3, fake2]))
                # video_array.append(cv2.vconcat([fake1, fake3, fake2]))
                # cv2.waitKey(10)
                # global_counter += 1
                if good_pts / total_pts < surface_conf: continue

                # update position of front point and base points
                drivable_pts.append(np.array([candx, y, candz, h]))
                drawable_pts.append(point_normal)
                drawable_pts_pt_cloud.append(point_pt_cld)
                base_pts[:] = []
                base_pts = new_base
                top_pts[:] = []
                top_pts = new_top
                num_of_pts += 1
                bbox3d_loc = sum(new_base)/len(new_base)
                bbox3d_loc = bbox3d_loc[:-1]

                # generate new 180 degree sweep angles for the next position
                # angles must not cross left and right limit because then predicted path
                # might take a u-turn as it looks for positions in the direction that is
                # opposite to the one the car is facing
                left_angles[:] = []
                right_angles[:] = []
                all_angles[:] = []
                _left_angles = [(angle+i*res) for i in range(int((np.pi/2) // res))
                               if (angle+i*res) < left_angle_limit]
                _right_angles = [(angle-i*res) for i in range(1,int((np.pi/2) // res))
                                if (angle-i*res) > right_angle_limit]
                left_angles = [i%(2*np.pi) for i in _left_angles]
                right_angles = [i%(2*np.pi) for i in _right_angles]
                all_angles = left_angles + right_angles
                got_pt = True
                break
            if got_pt: break
        all_steps = np.arange(0.1, 0.6, 0.1)

        # means no drivable area was found in the full 180 degree sweep
        if num_of_pts == num_of_pts_before: break

    # only draw when you have a path that is longer than the minimum path limit
    if len(drawable_pts) >= min_path_limit:
        got_path = True

        # draw line from starting position to position of predicted bounding box
        for i in range(len(drawable_pts)-1):
            pt1, pt2 = drawable_pts[i], drawable_pts[i+1]
            cv2.line(original_img, pt1, pt2, (255,0,255), 3)
            pt1, pt2 = drawable_pts_pt_cloud[i], drawable_pts_pt_cloud[i+1]
            cv2.line(pt_cloud_img, pt1, pt2, (255,0,255), 3)

        # plot 3d bounding box at new position
        plot_base_and_top(original_img, pt_cloud_img, base_pts, top_pts, calib, offsets)

    # for i in range(len(video_array)):
    #     out.write(video_array[i])
    # out.release()
    return drivable_pts, got_path

# bbox3d_roty must be between -pi and pi as is the case in the KITTI label format
def plot_3d_bbox(img, calib, bbox3d_center, bbox3d_dims, bbox3d_roty):
    box_3d = []
    box_pts = []
    h, w, l = bbox3d_dims
    p0, p1, p2, p3 = np.array([l/2,0,w/2]), np.array([-l/2,0,w/2]), 
                     np.array([-l/2,0,-w/2]), np.array([l/2,0,-w/2])
    p4, p5, p6, p7 = np.array([l/2,-h,w/2]), np.array([-l/2,-h,w/2]), 
                     np.array([-l/2,-h,-w/2]), np.array([l/2,-h,-w/2])
    pts_array = np.array([p0, p1, p2, p3, p4, p5, p6, p7]).transpose()
    rot_mat = np.array([[cos(bbox3d_roty), 0, sin(bbox3d_roty)],[0, 1, 0],[-sin(bbox3d_roty), 0, cos(bbox3d_roty)]])
    pts_array = np.matmul(rot_mat, pts_array).transpose()
    for pt_array in pts_array:
        box_pts.append(np.append(pt_array+bbox3d_center, 1))
        box_3d.append(get_img_pt(np.append(pt_array+bbox3d_center, 1), calib))
    for i in [0,1,2,3]:
        pt1, pt2, pt3, pt4 = box_3d[i%4], box_3d[(i+1)%4], box_3d[(i+4)%8], box_3d[(i+5)%8]
        pt5, pt6 = box_3d[(i%4)+4], box_3d[((i+1)%4)+4]
        cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        cv2.line(img, pt1, pt3, (0, 0, 255), 1)
        cv2.line(img, pt2, pt4, (0, 0, 255), 1)
        cv2.line(img, pt5, pt6, (0, 0, 255), 1)
    cv2.line(img, box_3d[0], box_3d[-1], (0, 0, 255), 1)
    cv2.line(img, box_3d[3], box_3d[4], (0, 0, 255), 1)
    center_pt_img = get_img_pt(np.append(bbox3d_center, 1), calib)
    cv2.circle(img, center_pt_img, 3, (255, 255, 255), -1)

    return box_pts

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

def calculate_iou(box_1, box_2):
    poly_1 = Polygon([[box_1[0], box_1[1]], [box_1[2], box_1[1]], [box_1[2], box_1[3]], [box_1[0], box_1[3]]])
    poly_2 = Polygon([[box_2[0], box_2[1]], [box_2[2], box_2[1]], [box_2[2], box_2[3]], [box_2[0], box_2[3]]])
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

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
                    default='/mnt/sdb1/datasets/kitti_tracking/training/0001', type=str,
                    help='your data folder location')
parser.add_argument('--velo_type', default='velodyne', type=str, help='choices: velodyne, fake_velodyne')
parser.add_argument('--operation_type', default='ROAD CLASSIFICATION', type=str,
                    help='choices: GATHER DATA, TRAJECTORY PRED AVERAGE 2D, '
                         'TRAJECTORY PRED 3D PROP, POTENTIAL FIELD, ROAD CLASSIFICATION, '
                         'SANITY')
args = parser.parse_args()

#folder locations
left_img_folder = os.path.join(args.data_loc, 'image_2')
calib_folder = os.path.join(args.data_loc, 'calib')
velo_folder = os.path.join(args.data_loc, args.velo_type)
label_folder = os.path.join(args.data_loc, 'label_2')

allowable_dets = ['Car', 'Pedestrian', 'Cyclist', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']

if args.operation_type in ['TRAJECTORY PRED 3D PROP', 'POTENTIAL FIELD', 'ROAD CLASSIFICATION']:
    pred_3dprop_folder = os.path.join(args.data_loc, '3DPROP')
    Path(pred_3dprop_folder).mkdir(parents=True, exist_ok=True)
    potential_folder = os.path.join(args.data_loc, 'POTENTIAL')
    Path(potential_folder).mkdir(parents=True, exist_ok=True)
    road_class_folder = os.path.join(args.data_loc, 'ROAD_CLASSIFIER')
    Path(road_class_folder).mkdir(parents=True, exist_ok=True)

    # get panop images
    panop_img_folder = os.path.join(args.data_loc, 'PANOP_IMGS')

    # get detections
    det_folder = os.path.join(args.data_loc, 'DETS3D')
    det_files = natsort.natsorted(os.listdir(det_folder))

    left_imgs = natsort.natsorted(os.listdir(left_img_folder))
    for left_img in left_imgs:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
        frame_num_str = left_img.split('.')[0]
        frame_num_int = int(frame_num_str)

        # get calibration parameters for the frame
        clb_file = open(os.path.join(calib_folder, frame_num_str+'.txt'))
        clb = {}
        for line in clb_file:
            calib_line = line.split(':')
            if len(calib_line) < 2:
                continue
            key = calib_line[0]
            value = np.array(list(map(float, calib_line[1].split())))
            value = value.reshape((3, -1))
            clb[key] = value

        # load image
        loaded_img = cv2.imread(os.path.join(left_img_folder, left_img))
        image_2d = np.copy(loaded_img)
        height, width, channels = loaded_img.shape

        # load detections
        if not os.path.exists(os.path.join(det_folder, frame_num_str+'.txt')) or 
                os.stat(os.path.join(det_folder, frame_num_str+'.txt')).st_size < 10:
            continue
        det = pd.read_csv(os.path.join(det_folder, frame_num_str+'.txt'), sep=" ", header=None)
        det = list(det.to_numpy())
        names, boxes = [], []
        for label_line in det:
            new_label_line = [x for x in label_line if str(x) != 'nan']
            if new_label_line[0] in allowable_dets:
                names.append(new_label_line[0])
                boxes.append(np.array(new_label_line[8:-1]))    # last element is score
        if not boxes:
            cv2.imshow('BBOXES', loaded_img)
            cv2.imwrite(os.path.join(pred_3dprop_folder, left_img), loaded_img)
            cv2.waitKey(1)
            continue

        # store all 3d detections in a dictionary
        if not os.path.exists(os.path.join(panop_img_folder, left_img)): continue
        panop_img = cv2.imread(os.path.join(panop_img_folder, left_img))
        panop_img = cv2.cvtColor(panop_img, cv2.COLOR_BGR2GRAY)
        all_dets = []
        for name, box in zip(names, boxes):
            bbox3d_roty = box[-1]
            dist = np.linalg.norm(box[3:6])
            all_dets.append({'name': name, 'roty': bbox3d_roty, 'bbox3d': box[:6], 'dist': dist})
            bbox3d_loc = box[3:6]
            bbox3d_dims = box[:3]
            # _ = plot_3d_bbox(loaded_img, clb, bbox3d_loc, bbox3d_dims, bbox3d_roty)
        # loaded_img = cv2.resize(loaded_img, (1280, 720))
        # cv2.imshow('', loaded_img)
        # cv2.waitKey(5000)
        # continue

        # load velodyne points
        if not os.path.exists(os.path.join(velo_folder, frame_num_str+'.bin')): continue
        velo_pts = np.fromfile(os.path.join(velo_folder, frame_num_str+'.bin'),
                               dtype=np.float32, count=-1).reshape([-1, 4])
        velo_pts = list(velo_pts)

        # draw point cloud from a top down perspective
        pt_cloud_img = np.zeros((height, width, channels), np.uint8)
        road_pts = []
        for v in velo_pts:
            pt2d_persp = diff_persp(v, clb, 5, 0, -2, [0,2], True)
            pt2d_normal, pt_in_cam = get_2d_pt_from_vel(v, clb)
            if pt2d_persp.size != 0 and 0 <= int(pt2d_persp[0][0]) < width and 0 <= int(pt2d_persp[0][1]) < height:
                pt2dint_persp = (int(pt2d_persp[0][0]), int(pt2d_persp[0][1]))
                cv2.circle(pt_cloud_img, pt2dint_persp, 1, (255, 255, 255), -1)

            if args.operation_type != 'ROAD CLASSIFICATION': continue
            if pt2d_normal.size != 0 and 0 <= int(pt2d_normal[0][0]) < width and pt_in_cam.size != 0 and 
                    0 <= int(pt2d_normal[0][1]) < height:
                pt2dint_normal = (int(pt2d_normal[0][0]), int(pt2d_normal[0][1]))
                if panop_img[pt2dint_normal[1], pt2dint_normal[0]] == 255:      # get all road points
                    road_pts.append({'pt': v, 'persp_pt': pt2dint_persp, 'norm_pt': pt2dint_normal})

        all_dets = sorted(all_dets, key=lambda k: k['dist'])
        if len(all_dets) >= 3: all_dets = all_dets[:3]
        object_iter = 0
        display_dict = {}
        sufficient_path = False

        # draw heatmap legend on image
        if args.operation_type == 'ROAD CLASSIFICATION':
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
                        if counter == 0: color_ind = 0
                        if counter == 15: color_ind = 0
                        if counter == 39: color_ind = -1
                        text_rgb = color_type[color_ind].rgb
                        text_rgb = (int(text_rgb[2]*256), int(text_rgb[1]*256), int(text_rgb[0]*256))
                        cv2.putText(loaded_img, text, (x_val, 100), font, 0.5, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(pt_cloud_img, text, (x_val, 100), font, 0.5, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(loaded_img, text, (x_val, 100), font, 0.5, text_rgb, 2, cv2.LINE_AA)
                        cv2.putText(pt_cloud_img, text, (x_val, 100), font, 0.5, text_rgb, 2, cv2.LINE_AA)
                    counter += 1

        # do road classification
        blank_image = np.zeros((height, width, channels), np.uint8)
        print(all_dets)
        while args.operation_type == 'ROAD CLASSIFICATION':
            theta = all_dets[object_iter]['roty']
            name = all_dets[object_iter]['name']
            bbox3d_dims, bbox3d_loc = all_dets[object_iter]['bbox3d'][:3], all_dets[object_iter]['bbox3d'][3:]

            # object facing away from you has negative rotation (anti-clockwise from positive x axis)
            # object facing towards you has positive rotation (clockwise from positive x axis)
            orient = abs(theta) if theta <= 0 else 2*np.pi - theta
            res = np.pi/180
            left_angles = [(orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res)+1)]
            right_angles = [(orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res)+1)]
            all_angles = left_angles + right_angles

            # create bounding box in the beginning
            offsets = [0, 2, 5]
            offsetx, offsety, offsetz = offsets
            points = plot_3d_bbox(np.copy(loaded_img), clb, bbox3d_loc, bbox3d_dims, all_dets[object_iter]['roty'])
            plot_base_and_top(loaded_img, pt_cloud_img, points[:4], points[4:], clb, offsets)

            temp_blank_image = np.zeros((height, width, channels), np.uint8)
            for angle in all_angles:
                generate_position_heatmap(panop_img, loaded_img, temp_blank_image,
                                          clb, angle, all_dets[object_iter])
            blank_image = blank_image + temp_blank_image
            display_dict['img'], display_dict['pt_img'] = loaded_img, pt_cloud_img
            if object_iter < len(all_dets) - 1: object_iter += 1
            else: break     # if you have arrived at the end of your object list, then leave

        # draw potential field
        while not sufficient_path and args.operation_type == 'POTENTIAL FIELD':
            theta = all_dets[object_iter]['roty']
            name = all_dets[object_iter]['name']

            # object facing away from you has negative rotation (anti-clockwise from positive x axis)
            # object facing towards you has positive rotation (clockwise from positive x axis)
            orient = abs(theta) if theta <= 0 else 2*np.pi - theta
            res = 0.3
            left_angles = [(orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res)+1)]
            right_angles = [(orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res)+1)]
            all_angles = left_angles + right_angles
            det_bbox3d_dims = all_dets[object_iter]['bbox3d'][:3]
            det_bbox3d_loc = all_dets[object_iter]['bbox3d'][3:]

            # get point on front-facing side of 3d bbox
            pts = plot_3d_bbox(np.copy(loaded_img), clb, det_bbox3d_loc, det_bbox3d_dims, theta)
            front_pts = [pts[0], pts[3]]
            object_loc = sum(front_pts) / len(front_pts)

            # create bounding box in the beginning
            bbox3d_dims, bbox3d_loc = all_dets[object_iter]['bbox3d'][:3], all_dets[object_iter]['bbox3d'][3:]
            offsets = [0, 2, 5]
            offsetx, offsety, offsetz = offsets
            points = plot_3d_bbox(np.copy(loaded_img), clb, bbox3d_loc, bbox3d_dims, all_dets[object_iter]['roty'])
            plot_base_and_top(loaded_img, pt_cloud_img, points[:4], points[4:], clb, offsets)

            boolean_list = []
            color_img = np.copy(loaded_img)
            pt_img = np.copy(pt_cloud_img)
            for angle in all_angles:
                got_path = generate_potential_field(object_loc, panop_img, color_img, pt_img, clb, angle,
                                                    all_dets[object_iter], det_bbox3d_dims[2])
                boolean_list.append(got_path)
                display_dict['img'], display_dict['pt_img'] = color_img, pt_img

            # check if you have at least one line in the potential field
            if sum(boolean_list) > 1:
                _ = plot_3d_bbox(color_img, clb, det_bbox3d_loc, det_bbox3d_dims, theta)
                break

            # if you did not get a potential field, then move on to the next detected object
            if object_iter < len(all_dets) - 1: object_iter += 1
            else: break     # if you have arrived at the end of your object list, then leave

        # propagate bounding box based on panoptic segmentation
        while not sufficient_path and args.operation_type == 'TRAJECTORY PRED 3D PROP':
            theta = all_dets[object_iter]['roty']
            name = all_dets[object_iter]['name']

            # object facing away from you has negative rotation (anti-clockwise from positive x axis)
            # object facing towards you has positive rotation (clockwise from positive x axis)
            orient = abs(theta) if theta <= 0 else 2*np.pi - theta
            res = 0.5
            left_angles = [(orient+i*res)%(2*np.pi) for i in range(int((np.pi/2) // res)+1)]
            right_angles = [(orient-i*res)%(2*np.pi) for i in range(1,int((np.pi/2) // res)+1)]
            left_angle_limit, right_angle_limit = left_angles[-1], right_angles[-1]
            all_angles = left_angles + right_angles

            det_bbox3d_dims = all_dets[object_iter]['bbox3d'][:3]

            for angle in all_angles:
                color_img = np.copy(loaded_img)
                pt_img = np.copy(pt_cloud_img)
                drivable_pts, sufficient_path = generate_path(panop_img, color_img, clb, angle,
                                                              all_dets[object_iter], sufficient_path,
                                                              2*det_bbox3d_dims[2], pt_img,
                                                              left_angle_limit, right_angle_limit)
                if sufficient_path:
                    display_dict['img'], display_dict['pt_img'], display_dict['drivable_pts'] = 
                        color_img, pt_img, len(drivable_pts)
                    break
            # if you did not get a predicted 3d boundin box, then move on to the next detected object
            if object_iter < len(all_dets) - 1: object_iter += 1
            else: break     # if you have arrived at the end of your object list, then leave
        if not display_dict:
            print(left_img)
            continue
        if args.operation_type in ['TRAJECTORY PRED 3D PROP', 'POTENTIAL FIELD']:
            color_img = display_dict['img']
            pt_img = display_dict['pt_img']

        # cd into folder containing images with trajectory predictions drawn on them
        # type following in console:
        # mencoder mf://*.png -mf fps=10:type=png -ovc x264 -x264encopts bitrate=1200:threads=2 -o outputfile.mkv
        if args.operation_type == 'TRAJECTORY PRED 3D PROP':
            cv2.imshow('BBOXES AND POINT CLOUD', cv2.vconcat([color_img, pt_img]))
            cv2.imwrite(os.path.join(pred_3dprop_folder, left_img), cv2.vconcat([color_img, pt_img]))
        if args.operation_type == 'POTENTIAL FIELD':
            cv2.imshow('BBOXES AND POINT CLOUD', cv2.vconcat([color_img, pt_img]))
            cv2.imwrite(os.path.join(potential_folder, left_img), cv2.vconcat([color_img, pt_img]))
        if args.operation_type == 'ROAD CLASSIFICATION':
            loaded_img = cv2.addWeighted(blank_image, 0.5, loaded_img, 1.0, 0)
            for road_pt in road_pts:
                normal_pixel_loc, persp_pixel_loc = road_pt['norm_pt'], road_pt['persp_pt']
                if (blank_image[normal_pixel_loc[1], normal_pixel_loc[0]] != [0,0,0]).any():
                    color_val = blank_image[normal_pixel_loc[1], normal_pixel_loc[0]]
                    color_val = (int(color_val[0]), int(color_val[1]), int(color_val[2]))
                    cv2.circle(pt_cloud_img, persp_pixel_loc, 3, color_val, -1)
            cv2.imshow('BBOXES AND POINT CLOUD', cv2.vconcat([loaded_img, pt_cloud_img]))
            cv2.imwrite(os.path.join(road_class_folder, left_img), cv2.vconcat([loaded_img, pt_cloud_img]))
        print(left_img)
        cv2.waitKey(1)

if args.operation_type == 'TRAJECTORY PRED AVERAGE 2D':
    pred_avg2d_folder = os.path.join(args.data_loc, 'AVG2D')
    Path(pred_avg2d_folder).mkdir(parents=True, exist_ok=True)

    #for average prediction
    obj_dict = {}
    mot_tracker = Sort()
    input_data_length = 5
    predicted_data_length = 10
    max_diag = 100

    #get detections
    det_folder = os.path.join(args.data_loc, 'DETS2D')
    det_files = natsort.natsorted(os.listdir(det_folder))

    left_imgs = natsort.natsorted(os.listdir(left_img_folder))
    for left_img in left_imgs:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
        frame_num_str = left_img.split('.')[0]
        frame_num_int = int(frame_num_str)

        # load image
        loaded_img = cv2.imread(os.path.join(left_img_folder, left_img))
        image_2d = np.copy(loaded_img)
        height, width, channels = loaded_img.shape

        # load detection
        if not os.path.exists(os.path.join(det_folder, frame_num_str+'.txt')): continue
        det = pd.read_csv(os.path.join(det_folder, frame_num_str+'.txt'), sep=" ", header=None)
        det = list(det.to_numpy())
        names, boxes = [], []
        for label_line in det:
            if label_line[0] in allowable_dets:
                names.append(label_line[0])
                boxes.append(np.array(label_line[4:8]))
        if not boxes:
            cv2.imshow('AVERAGE BBOXES', loaded_img)
            cv2.imwrite(os.path.join(pred_avg2d_folder, left_img), loaded_img)
            cv2.waitKey(1)
            continue

        # track bboxes
        det_ids = []
        tracked_bbs = mot_tracker.update(np.array(boxes))
        for bbs in tracked_bbs:
            if bbs[-1] in obj_dict: obj_dict[bbs[-1]].append(bbs[:-1])
            else: obj_dict[bbs[-1]] = []
            det_ids.append(bbs[-1])

        for obj_id, obj_bbox_list in obj_dict.items():
            # only show id detected by tracker
            if obj_id not in det_ids: continue

            # only display bboxes of certain size (less clutter on screen)
            diag_len = 0
            if obj_bbox_list:
                p1 = np.array([int(obj_bbox_list[-1][0]), int(obj_bbox_list[-1][1])])
                p2 = np.array([int(obj_bbox_list[-1][2]), int(obj_bbox_list[-1][3])])
                diag_len = np.linalg.norm(p1-p2)

            # start prediction only when you have sufficient amount of data
            if len(obj_bbox_list) < input_data_length or diag_len < max_diag:
                continue

            # plot current detected bbox
            p1 = (int(obj_bbox_list[-1][0]), int(obj_bbox_list[-1][1]))
            p2 = (int(obj_bbox_list[-1][2]), int(obj_bbox_list[-1][3]))
            cv2.rectangle(loaded_img, p1, p2, (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(loaded_img,str(obj_id),(p1[0],p1[1]-20),font,1,(255,0,0),2,cv2.LINE_AA)

            # do prediction based on average differences
            avg_diff = np.zeros(4)
            for i in range(-input_data_length,-1):
                avg_diff += obj_bbox_list[i+1] - obj_bbox_list[i]
            avg_diff = avg_diff / input_data_length

            # plot predicted bbox
            p1 = (int(obj_bbox_list[-1][0] + predicted_data_length*avg_diff[0]),
                  int(obj_bbox_list[-1][1] + predicted_data_length*avg_diff[1]))
            p2 = (int(obj_bbox_list[-1][2] + predicted_data_length*avg_diff[2]),
                  int(obj_bbox_list[-1][3] + predicted_data_length*avg_diff[3]))
            cv2.rectangle(loaded_img, p1, p2, (0,0,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(loaded_img,str(obj_id),(p1[0],p1[1]-20),font,1,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow('AVERAGE BBOXES', loaded_img)
        cv2.imwrite(os.path.join(pred_avg2d_folder, left_img), loaded_img)
        cv2.waitKey(1)

if args.operation_type == 'GATHER DATA':
    for allowable_det in allowable_dets:
        img_data_folder = os.path.join(args.data_loc, allowable_det, 'IMAGES')
        velo_data_folder = os.path.join(args.data_loc, allowable_det, 'VELO')
        gt_folder = os.path.join(args.data_loc, allowable_det, 'GT')
        Path(gt_folder).mkdir(parents=True, exist_ok=True)
        Path(img_data_folder).mkdir(parents=True, exist_ok=True)
        Path(velo_data_folder).mkdir(parents=True, exist_ok=True)

    # label must follow KITTI Object Detection format
    # [name truncated occluded alpha
    # 2dbbox_x1 2d_bbox_y1 2d_bbox_x2 2d_bbox_y2
    # 3d_bbox_height 3d_bbox_width 3d_bbox_length 3d_bbox_x 3d_bbox_y 3d_bbox_z rot_y]
    frame_dict = {}
    ignore_frame = []
    label_files = natsort.natsorted(os.listdir(label_folder))
    for label_file in label_files:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
        if os.stat(os.path.join(label_folder, label_file)).st_size == 0:
            ignore_frame.append(label_file.split('.')[0])
            continue
        data = pd.read_csv(os.path.join(label_folder, label_file), sep=" ", header=None)
        data = list(data.to_numpy())
        all_valid_dets = []
        for label in data:
            if label[0] in allowable_dets:
                all_valid_dets.append(label)
        frame_num = label_file.split('.')[0]
        frame_dict[frame_num] = all_valid_dets

    #load model for instance segmentation
    config_file_inst = 'DetectoRS/configs/DetectoRS/DetectoRS_mstrain_400_1200_x101_32x4d_40e.py'
    checkpoint_file_inst = 'DetectoRS/DetectoRS_X101-ed983634.pth'
    model_inst = init_detector(config_file_inst, checkpoint_file_inst, device='cuda:0')

    left_imgs = natsort.natsorted(os.listdir(left_img_folder))
    obj_counter_car = 0
    obj_counter_cyclist = 0
    obj_counter_pedestrian = 0
    for left_img in left_imgs:      #MAKE CHANGES HERE TO ONLY GO THROUGH SOME IMAGES
        frame_num_str = left_img.split('.')[0]
        frame_num_int = int(frame_num_str)
        if frame_num_str in ignore_frame: continue

        # get calibration parameters for the frame
        clb_file = open(os.path.join(calib_folder, frame_num_str+'.txt'))
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
        if not os.path.exists(os.path.join(velo_folder, frame_num_str+'.bin')): continue
        velo_pts = np.fromfile(os.path.join(velo_folder, frame_num_str+'.bin'),
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
            if det_name not in allowable_dets:
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
                pt2d, _ = get_2d_pt_from_vel(v, clb)
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
            obj_counter = 0
            if gt_name == 'Car':
                obj_counter = obj_counter_car
                obj_counter_car += 1
            if gt_name == 'Cyclist':
                obj_counter = obj_counter_cyclist
                obj_counter_cyclist += 1
            if gt_name == 'Pedestrian':
                obj_counter = obj_counter_pedestrian
                obj_counter_pedestrian += 1

            # store velo data for training 3d object detector
            img_data_folder = os.path.join(args.data_loc, gt_name, 'IMAGES')
            velo_data_folder = os.path.join(args.data_loc, gt_name, 'VELO')
            gt_folder = os.path.join(args.data_loc, gt_name, 'GT')
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
            gt_file.write('
')
            gt_file.close()

        print("PROCESSED IMAGE " + left_img)
        print("CAR COUNTER SO FAR: " + str(obj_counter_car))
        print("CYCLIST COUNTER SO FAR: " + str(obj_counter_cyclist))
        print("PEDESTRIAN COUNTER SO FAR: " + str(obj_counter_pedestrian))


if args.operation_type == 'SANITY':
    original_imgs_folder = os.path.join(args.data_loc, 'image_2')
    calib_files_folder = os.path.join(args.data_loc, 'calib')
    original_imgs = natsort.natsorted(os.listdir(original_imgs_folder))
    calib_files = natsort.natsorted(os.listdir(calib_files_folder))
    DET_TYPES = ['Car', 'Cyclist', 'Pedestrian']

    for DET in DET_TYPES:
        image_folder = os.path.join(args.data_loc, DET, 'IMAGES')
        velo_folder = os.path.join(args.data_loc, DET, 'VELO')
        gt_folder = os.path.join(args.data_loc, DET, 'GT')

        images = natsort.natsorted(os.listdir(image_folder))
        velo_files = natsort.natsorted(os.listdir(velo_folder))
        gt_files = natsort.natsorted(os.listdir(gt_folder))
        c = list(zip(images, velo_files, gt_files))
        random.shuffle(c)
        images, velo_files, gt_files = zip(*c)

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
                pt2d, _ = get_2d_pt_from_vel(v, clb)
                pt2dint = (int(pt2d[0][0]), int(pt2d[0][1]))
                cv2.circle(partial_pt_cloud_img, (pt2dint[0], pt2dint[1]), 1, 255, -1)

            pt1 = (int(gt_bbox2d[0]), int(gt_bbox2d[1]))
            pt2 = (int(gt_bbox2d[2]), int(gt_bbox2d[3]))
            cv2.rectangle(original_img2d, pt1, pt2, (255, 0, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(original_img2d, gt_name, (pt1[0], pt1[1]-20), font, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('2D BBOX', original_img2d)
            _ = plot_3d_bbox(original_img3d, clb, gt_bbox3d_loc, gt_bbox3d_dims, gt_roty)
            cv2.imshow('3D BBOX', original_img3d)
            cv2.imshow('PARTIAL POINT CLOUD', partial_pt_cloud_img)
            cv2.imshow('IMAGE CROP', crop_img)
            cv2.waitKey(5000)
