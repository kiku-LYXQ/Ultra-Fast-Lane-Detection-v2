
from data.dataloader import get_test_loader
from evaluation.tusimple.lane2 import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os, json, torch, scipy
import numpy as np
import platform
from scipy.optimize import leastsq
from data.constant import culane_col_anchor, culane_row_anchor

def generate_lines(out, out_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    grid = torch.arange(out.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out.softmax(1) * grid).sum(1) 
    
    loc = loc / (out.shape[1]-1) * 1640
    # n, num_cls, num_lanes
    valid = out_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in [1,2]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( loc[j,k,i] , culane_row_anchor[k] * 590))
                    fp.write('\n')

def generate_lines_col(out_col,out_col_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):
    
    grid = torch.arange(out_col.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out_col.softmax(1) * grid).sum(1) 
    
    loc = loc / (out_col.shape[1]-1) * 590
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            for i in [0,3]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( culane_col_anchor[k] * 1640, loc[j,k,i] ))
                    fp.write('\n')

def generate_lines_local(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 

                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_local(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_local_curve_combine(dataset, out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [1, 2]
        elif dataset == 'CurveLanes':
            # lane_list = [2, 3, 4, 5, 6, 7]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out.shape[1]-1) * 1640
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out.shape[1]-1) * 2560
                                fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 1440))
                            else:
                                raise Exception
                    fp.write('\n')
                else:
                    fp.write('\n')

def generate_lines_col_local_curve_combine(dataset, out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        if dataset == 'CULane':
            lane_list = [0, 3]
        elif dataset == 'CurveLanes':
            # lane_list = [0, 1, 8, 9]
            lane_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):
        # import pdb; pdb.set_trace()

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            if dataset == 'CULane':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                            elif dataset == 'CurveLanes':
                                out_tmp = out_tmp / (out_col.shape[1]-1) * 1440
                                fp.write('%.3f %.3f '% ( col_anchor[k] * 2560, out_tmp ))
                            else:
                                raise Exception

                    fp.write('\n')
                # elif mode == 'all':
                #     fp.write('\n')
                else:
                    fp.write('\n')

def revise_lines_curve_combine(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        row_line_save_path = os.path.join(output_path, name[:-3] + 'lines_row.txt')
        col_line_save_path = os.path.join(output_path, name[:-3] + 'lines_col.txt')
        if not os.path.exists(row_line_save_path):
            continue
        if not os.path.exists(col_line_save_path):
            continue
        with open(row_line_save_path, 'r') as fp:
            row_lines = fp.readlines()
        with open(col_line_save_path, 'r') as fp:
            col_lines = fp.readlines()
        flag = True
        for i in range(10):
            x1, y1 = coordinate_parse(row_lines[i])
            x2, y2 = coordinate_parse(col_lines[i])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 36)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()

def generate_lines_reg(out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu().sigmoid()

    if mode == 'normal' or mode == '2row2col':
        lane_list = [1, 2]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = out[j,0,k,i] * 1640

                            fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_reg(out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    # max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu().sigmoid()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        lane_list = [0, 3]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            # out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_col[j,0,k,i] * 590
                            fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def coordinate_parse(line):
    if line == '\n':
        return [], []

    items = line.split(' ')[:-1]
    x = [float(items[2*i]) for i in range(len(items)//2)]
    y = [float(items[2*i+1]) for i in range(len(items)//2)]

    return x, y


def func(p, x):
    f = np.poly1d(p)
    return f(x)


def resudual(p, x, y):
    error = y - func(p, x)
    return error


def revise_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for i in range(4):
            x1, y1 = coordinate_parse(lines[i])
            x2, y2 = coordinate_parse(lines[i+4])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()
            

def rectify_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for line in lines:
            x, y = coordinate_parse(line)
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()


def run_test(dataset, net, data_root, exp_name, work_dir, distributed, crop_ratio, train_width, train_height , batch_size=8, row_anchor = None, col_anchor = None):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            pred = net(imgs)
        
        if dataset == "CULane":
            generate_lines_local(dataset, pred['loc_row'],pred['exist_row'], names, output_path, 'normal', row_anchor=row_anchor)
            generate_lines_col_local(dataset, pred['loc_col'],pred['exist_col'], names, output_path, 'normal', col_anchor=col_anchor)
        elif dataset == 'CurveLanes':
            generate_lines_local_curve_combine(dataset, pred['loc_row'],pred['exist_row'], names, output_path, row_anchor=row_anchor)
            generate_lines_col_local_curve_combine(dataset, pred['loc_col'],pred['exist_col'], names, output_path, col_anchor=col_anchor)
            revise_lines_curve_combine(names, output_path)
        else:
            raise NotImplementedError



def generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor):

    local_width = 1

    max_indices = loc_row.argmax(1).cpu()
    valid = exist_row.argmax(1).cpu()
    loc_row = loc_row.cpu()

    max_indices_left = loc_row_left.argmax(1).cpu()
    valid_left = exist_row_left.argmax(1).cpu()
    loc_row_left = loc_row_left.cpu()

    max_indices_right = loc_row_right.argmax(1).cpu()
    valid_right = exist_row_right.argmax(1).cpu()
    loc_row_right = loc_row_right.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_row.shape

    min_lane_length = num_cls / 2

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg', 'lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [1,2]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_row[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 1640
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_left[batch_idx,cls_idx,lane_idx]:
                            all_ind_left = torch.tensor(list(range(max(0,max_indices_left[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_left[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_left = (loc_row_left[batch_idx,all_ind_left,cls_idx,lane_idx].softmax(0) * all_ind_left.float()).sum() + 0.5 
                            out_tmp_left = out_tmp_left / (num_grid-1) * 1640 + 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_left

                        if valid_right[batch_idx,cls_idx,lane_idx]:
                            all_ind_right = torch.tensor(list(range(max(0,max_indices_right[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_right[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_right = (loc_row_right[batch_idx,all_ind_right,cls_idx,lane_idx].softmax(0) * all_ind_right.float()).sum() + 0.5 
                            out_tmp_right = out_tmp_right / (num_grid-1) * 1640 - 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_right


                        if cnt >= 2:
                            pt_all.append(( out_tmp_all/cnt , row_anchor[cls_idx] * 590))
                    if len(pt_all) < min_lane_length:
                            continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor):
    local_width = 1
    
    max_indices = loc_col.argmax(1).cpu()
    valid = exist_col.argmax(1).cpu()
    loc_col = loc_col.cpu()

    max_indices_up = loc_col_up.argmax(1).cpu()
    valid_up = exist_col_up.argmax(1).cpu()
    loc_col_up = loc_col_up.cpu()

    max_indices_down = loc_col_down.argmax(1).cpu()
    valid_down = exist_col_down.argmax(1).cpu()
    loc_col_down = loc_col_down.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_col.shape

    min_lane_length = num_cls / 4

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg','lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [0,3]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_col[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_up[batch_idx,cls_idx,lane_idx]:
                            all_ind_up = torch.tensor(list(range(max(0,max_indices_up[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_up[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_up = (loc_col_up[batch_idx,all_ind_up,cls_idx,lane_idx].softmax(0) * all_ind_up.float()).sum() + 0.5 
                            out_tmp_up = out_tmp_up / (num_grid-1) * 590 + 32./534*590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_up
                        if valid_down[batch_idx,cls_idx,lane_idx]:
                            all_ind_down = torch.tensor(list(range(max(0,max_indices_down[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_down[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_down = (loc_col_down[batch_idx,all_ind_down,cls_idx,lane_idx].softmax(0) * all_ind_down.float()).sum() + 0.5 
                            out_tmp_down = out_tmp_down / (num_grid-1) * 590 - 32./534*590     
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_down

                        if cnt >= 2:
                            pt_all.append(( col_anchor[cls_idx] * 1640, out_tmp_all/cnt ))
                    if len(pt_all) < min_lane_length:
                        continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def run_test_tta(dataset, net, data_root, exp_name, work_dir,distributed, crop_ratio, train_width, train_height, batch_size=8, row_anchor = None, col_anchor = None):
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()
    loader = get_test_loader(batch_size, data_root, dataset, distributed, crop_ratio, train_width, train_height)
    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            if hasattr(net, 'module'):
                pred = net.module.forward_tta(imgs)
            else:
                pred = net.forward_tta(imgs)

            loc_row, loc_row_left, loc_row_right, _, _ = torch.chunk(pred['loc_row'], 5)
            loc_col, _, _, loc_col_up, loc_col_down = torch.chunk(pred['loc_col'], 5)

            exist_row, exist_row_left, exist_row_right, _, _ = torch.chunk(pred['exist_row'], 5)
            exist_col, _, _, exist_col_up, exist_col_down = torch.chunk(pred['exist_col'], 5)


        generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor)
        generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor)


def generate_tusimple_lines(row_out, row_ext, col_out, col_ext, row_anchor=None, col_anchor=None, mode='2row2col'):
    """
    生成符合Tusimple数据集格式的车道线坐标

    参数:
        参数:
        row_out: 行方向预测输出（形状：[row_num_grid, row_num_cls, row_num_lane]）
            - row_num_grid: 横向网格数量（如100）
            - row_num_cls: <span style="color: red">纵向锚点数量（如56）
            - row_num_lane: 车道线数量（如4）
        row_ext: 行方向存在性概率<span style="color: red">（形状：[row_num_grid, row_num_cls, row_num_lane]）
        col_out: 列方向预测输出（形状：[col_num_grid, col_num_cls, col_num_lane]）
            - col_num_grid: 纵向网格数量
            - col_num_cls: 列锚点类别数（横向锚点数，如41）
            - col_num_lane: 车道线数量
        col_ext: 列方向存在性概率
        row_anchor: 纵向锚点的归一化位置（长度row_num_cls的数组）
        col_anchor: 横向锚点的归一化位置（长度col_num_cls的数组）
        mode: 处理模式，可选：
            - '2row2col'：处理2条行车道线和2条列车道线（默认）
            - '4row'：处理所有4条行车道线
            - '4col'：处理所有4条列车道线

    返回:
        all_lanes: 车道线坐标列表，每个元素是长度为56的列表，对应tusimple_h_sample的X坐标（-2表示无效点）
    """
    # Tusimple官方要求的固定Y轴采样点（56个点，从160到710像素）
    tusimple_h_sample = np.linspace(160, 710, 56)

    # 解析行预测输出的维度信息
    row_num_grid, row_num_cls, row_num_lane = row_out.shape  # 格式：[网格数, 锚点数, 车道数]
    # 获取行方向预测的最大概率索引（形状：[row_num_cls, row_num_lane]）
    row_max_indices = row_out.argmax(0).cpu()
    # 行存在性判断：在网格维度取argmax（若存在性预测是二分类，需调整维度顺序）
    row_valid = row_ext.argmax(0).cpu()  # 形状：[row_num_cls, row_num_lane]
    # 将数据移到CPU并转换为float类型
    row_out = row_out.cpu()

    # 解析列预测输出的维度信息（逻辑同行预测）
    col_num_grid, col_num_cls, col_num_lane = col_out.shape
    col_max_indices = col_out.argmax(0).cpu()
    col_valid = col_ext.argmax(0).cpu()
    col_out = col_out.cpu()

    # 根据模式选择要处理的车道线索引 --------------------------------------------------------
    if mode == 'normal' or mode == '2row2col':
        # 默认模式：处理第1、2条行车道线（索引从0开始）和第0、3条列车道线
        row_lane_list = [1, 2]  # 行车道线索引
        col_lane_list = [0, 3]  # 列车道线索引
    elif mode == '4row':
        # 处理所有4条行车道线
        row_lane_list = range(row_num_lane)
        col_lane_list = []
    elif mode == '4col':
        # 处理所有4条列车道线
        row_lane_list = []
        col_lane_list = range(col_num_lane)
    else:
        raise NotImplementedError(f"未实现的模式: {mode}")

    # 局部窗口参数配置 -----------------------------------------------------------------
    local_width_row = 14  # 行方向预测时考虑的局部窗口宽度（左右各扩展14个网格）
    local_width_col = 14  # 列方向预测时考虑的局部窗口宽度
    min_lanepts_row = 3  # 行车道线有效的最小点数阈值
    min_lanepts_col = 3  # 列车道线有效的最小点数阈值

    all_lanes = []  # 存储最终生成的所有车道线坐标

    # ============================================================================
    # 处理行车道线（垂直方向车道线，如左右车道线）
    # ============================================================================
    for row_lane_idx in row_lane_list:
        # 检查当前车道线是否存在：有效点数超过阈值
        if row_valid[:, row_lane_idx].sum() > min_lanepts_row:
            cur_lane = []  # 存储当前车道线在56个固定Y位置的X坐标

            # 遍历所有行锚点（row_num_cls=56个纵向位置）
            for row_cls_idx in range(row_num_cls):
                # 如果当前锚点位置存在车道线
                if row_valid[row_cls_idx, row_lane_idx]:
                    # 确定局部窗口范围（防止越界）
                    start_idx = max(0, row_max_indices[row_cls_idx, row_lane_idx] - local_width_row)
                    end_idx = min(row_num_grid - 1, row_max_indices[row_cls_idx, row_lane_idx] + local_width_row)
                    all_ind = torch.tensor(list(range(start_idx, end_idx + 1)))

                    # 计算加权平均位置（亚像素细化）
                    # 1. 对窗口内的预测值做softmax归一化
                    # 2. 用概率值加权求和得到精细化的网格位置
                    coord = (row_out[all_ind, row_cls_idx, row_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    # 将网格位置映射到实际图像坐标（假设输入图像尺寸为1280x720）
                    coord_x = coord / (row_num_grid - 1) * 1280  # <span style="color: red">横向坐标映射</span>
                    coord_y = row_anchor[row_cls_idx] * 720  # <span style="color: red">纵向坐标通过锚点比例计算</span>
                    cur_lane.append(int(coord_x))  # 仅记录X坐标，Y坐标由tusimple_h_sample固定
                else:
                    cur_lane.append(-2)  # -2表示该Y位置无有效点

            all_lanes.append(cur_lane)  # 添加当前车道线到结果列表
        else:
            # 如果有效点数不足，跳过该车道线
            pass

    # ============================================================================
    # 处理列车道线（水平方向车道线，如远端的横向车道线）
    # ============================================================================
    for col_lane_idx in col_lane_list:
        # 检查当前车道线是否存在：有效点数超过阈值
        if col_valid[:, col_lane_idx].sum() > min_lanepts_col:
            cur_lane = []  # 存储当前车道线的原始坐标点（用于曲线拟合）

            # 遍历所有列锚点（col_num_cls=41个横向位置）
            for col_cls_idx in range(col_num_cls):
                # 如果当前锚点位置存在车道线
                if col_valid[col_cls_idx, col_lane_idx]:
                    # 确定局部窗口范围（逻辑同行处理）
                    start_idx = max(0, col_max_indices[col_cls_idx, col_lane_idx] - local_width_col)
                    end_idx = min(col_num_grid - 1, col_max_indices[col_cls_idx, col_lane_idx] + local_width_col)
                    all_ind = torch.tensor(list(range(start_idx, end_idx + 1)))

                    # 计算加权平均位置（亚像素细化）
                    coord = (col_out[all_ind, col_cls_idx, col_lane_idx].softmax(0) * all_ind.float()).sum() + 0.5
                    coord_y = coord / (col_num_grid - 1) * 720  # 映射到Y轴
                    coord_x = col_anchor[col_cls_idx] * 1280  # 获取X轴实际坐标
                    cur_lane.append((coord_x, coord_y))  # 存储(X,Y)坐标对

            # 将原始点转换为numpy数组并进行二次多项式拟合 --------------------------------
            cur_lane = np.array(cur_lane)
            if len(cur_lane) >= 2:  # 至少需要2个点才能拟合
                top_lim = min(cur_lane[:, 1])  # 车道线最顶端Y坐标
                bot_lim = max(cur_lane[:, 1])  # 车道线最底端Y坐标

                # 使用二次多项式拟合X = f(Y)
                p = np.polyfit(cur_lane[:, 1], cur_lane[:, 0], deg=2)
                # 在固定Y采样点上计算X坐标
                lanes_on_tusimple = np.polyval(p, tusimple_h_sample)

                # 后处理：限制坐标范围和有效性
                lanes_on_tusimple = np.round(lanes_on_tusimple).astype(int)
                lanes_on_tusimple[lanes_on_tusimple < 0] = -2  # 过滤负值
                lanes_on_tusimple[lanes_on_tusimple > 1280] = -2  # 过滤超出右边界的值
                lanes_on_tusimple[tusimple_h_sample < top_lim] = -2  # 过滤顶端以上的点
                lanes_on_tusimple[tusimple_h_sample > bot_lim] = -2  # 过滤底端以下的点
                all_lanes.append(lanes_on_tusimple.tolist())
        else:
            # 如果有效点数不足，跳过该车道线
            pass

    return all_lanes  # 返回所有车道线的坐标列表


def run_test_tusimple(net, data_root, work_dir, exp_name, distributed, crop_ratio,
                      train_width, train_height, batch_size=8, row_anchor=None, col_anchor=None):
    """在Tusimple数据集上运行测试并生成评估文件

    参数:
        net: 训练好的车道线检测模型
        data_root: 数据集根目录路径
        work_dir: 工作目录，用于保存临时结果
        exp_name: 实验名称标识符（用于生成结果文件名）
        distributed: 是否使用分布式模式
        crop_ratio: 图像裁剪比例（数据增强参数）
        train_width: 训练时图像宽度（预处理尺寸）
        train_height: 训练时图像高度（预处理尺寸）
        batch_size: 批处理大小（默认8）
        row_anchor: 纵向锚点配置（Y轴采样位置）
        col_anchor: 横向锚点配置（X轴采样位置）
    """
    # 生成分布式环境下的结果文件路径（每个进程独立文件避免写入冲突）
    output_path = os.path.join(work_dir, exp_name + '.%d.txt' % get_rank())

    # 打开结果文件准备写入（文件格式需符合Tusimple官方评估要求）
    with open(output_path, 'w') as fp:
        # 获取测试集数据加载器（包含预处理和数据增强）
        loader = get_test_loader(
            batch_size, data_root, 'Tusimple', distributed,
            crop_ratio, train_width, train_height
        )

        # 使用分布式进度条遍历测试集（dist_tqdm为支持分布式的进度条工具）
        for data in dist_tqdm(loader):
            # 解构数据：imgs为预处理后的图像张量，names为对应的原始文件名
            imgs, names = data

            # 将图像数据移至GPU
            imgs = imgs.cuda()

            # 禁用梯度计算以提升推理效率
            with torch.no_grad():
                # 模型前向传播，获取预测结果
                # pred字典结构示例:
                # {
                #   'loc_row': 行方向坐标预测（形状：[B, 56, 4, 100]）
                #   - B: 批次大小
                #   - 56: 纵向锚点数(num_row)
                #   - 4: 车道线数量(num_lanes)
                #   - 100: 横向网格数(griding_num)
                #
                #   'exist_row': 行存在性概率（形状：[B, 2, 56, 4]）
                #   - 2: 存在性分类数（0-不存在，1-存在）
                #   - 56: 纵向锚点数
                #   - 4: 车道线数量
                #
                #   'loc_col': 列方向坐标预测（形状：[B, 41, 4, 100]）
                #   - 41: 横向锚点数(num_col)
                #   - 4: 车道线数量
                #   - 100: 纵向网格数
                #
                #   'exist_col': 列存在性概率（形状：[B, 2, 41, 4]）
                #   - 2: 存在性分类数
                #   - 41: 横向锚点数
                #   - 4: 车道线数量
                # }
                pred = net(imgs)

            # 遍历批次中的每个样本
            for b_idx, name in enumerate(names):
                # 构建符合Tusimple评估要求的JSON结构
                result_dict = {
                    # 生成车道线坐标（核心逻辑）
                    # generate_tusimple_lines参数说明:
                    # - loc_row: 行方向坐标预测（形状：[56, 4, 100]）
                    #   - 56: 纵向锚点数(num_row)
                    #   - 4: 车道线数量(num_lanes)
                    #   - 100: 横向网格数(griding_num)
                    # - exist_row: 行存在性概率（形状：[2, 56, 4]）
                    #   - 2: 存在性分类维度
                    #   - 56: 纵向锚点数
                    #   - 4: 车道线数量
                    # - loc_col: 列方向坐标预测（形状：[41, 4, 100]）
                    #   - 41: 横向锚点数(num_col)
                    #   - 4: 车道线数量
                    #   - 100: 纵向网格数
                    # - exist_col: 列存在性概率（形状：[2, 41, 4]）
                    #   - 2: 存在性分类维度
                    #   - 41: 横向锚点数
                    #   - 4: 车道线数量
                    # - mode='4row': 使用4行锚点解码模式
                    # - row_anchor: 纵向锚点位置（56个归一化坐标）
                    # - col_anchor: 横向锚点位置（41个归一化坐标）
                    'lanes': generate_tusimple_lines(
                        pred['loc_row'][b_idx],
                        pred['exist_row'][b_idx],
                        pred['loc_col'][b_idx],
                        pred['exist_col'][b_idx],
                        row_anchor=row_anchor,
                        col_anchor=col_anchor,
                        mode='4row'
                    ),

                    # Tusimple官方要求的固定Y轴采样点（从160到710，间隔10像素）
                    'h_samples': [i for i in range(160, 711, 10)],

                    # 原始图像文件名（需与标注文件对应）
                    'raw_file': name,

                    # 运行时间（单位ms，示例值，实际应记录真实推理时间）
                    'run_time': 10
                }

                # 将结果转换为JSON字符串并写入文件
                json_str = json.dumps(result_dict)
                fp.write(json_str + '\n')

    # 文件自动关闭（with语句保证）

def combine_tusimple_test(work_dir,exp_name):
    size = get_world_size()
    all_res = []
    for i in range(size):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% i)
        with open(output_path, 'r') as fp:
            res = fp.readlines()
        all_res.extend(res)
    names = set()
    all_res_no_dup = []
    for i, res in enumerate(all_res):
        pos = res.find('clips')
        name = res[pos:].split('\"')[0]
        if name not in names:
            names.add(name)
            all_res_no_dup.append(res)

    output_path = os.path.join(work_dir,exp_name+'.txt')
    with open(output_path, 'w') as fp:
        fp.writelines(all_res_no_dup)
    

def eval_lane(net, cfg, ep = None, logger = None):
    net.eval()
    if cfg.dataset == 'CurveLanes':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir, cfg.distributed,  cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()   # wait for all results
        if is_main_process():
            res = call_curvelane_eval(cfg.data_root, 'curvelanes_eval_tmp', cfg.test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            dist_print(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'CULane':
        if not cfg.tta:
            run_test(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        else:
            run_test_tta(cfg.dataset, net, cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()    # wait for all results
        if is_main_process():
            res = call_culane_eval(cfg.data_root, 'culane_eval_tmp', cfg.test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                dist_print(k,val)
                if logger is not None:
                    if k == 'res_cross':
                        logger.add_scalar('CuEval_cls/'+k,val_fp,global_step = ep)
                        continue
                    logger.add_scalar('CuEval_cls/'+k,val,global_step = ep)
            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            dist_print(F)
            if logger is not None:
                logger.add_scalar('CuEval/total',F,global_step = ep)
                logger.add_scalar('CuEval/P',P,global_step = ep)
                logger.add_scalar('CuEval/R',R,global_step = ep)
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None
    elif cfg.dataset == 'Tusimple':
        exp_name = 'tusimple_eval_tmp'
        run_test_tusimple(net, cfg.data_root, cfg.test_work_dir, exp_name, cfg.distributed, cfg.crop_ratio, cfg.train_width, cfg.train_height, row_anchor = cfg.row_anchor, col_anchor = cfg.col_anchor)
        synchronize()  # wait for all results
        if is_main_process():
            combine_tusimple_test(cfg.test_work_dir,exp_name)
            res = LaneEval.bench_one_submit(os.path.join(cfg.test_work_dir,exp_name + '.txt'),os.path.join(cfg.data_root,'test_label.json'))
            res = json.loads(res)
            for r in res:
                dist_print(r['name'], r['value'])
                if logger is not None:
                    logger.add_scalar('TuEval/'+r['name'],r['value'],global_step = ep)
        synchronize()
        if is_main_process():
            for r in res:
                if r['name'] == 'F1':
                    return r['value']
        else:
            return None


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval(data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=30
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir,'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir,'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir,'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir,'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir,'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir,'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir,'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir,'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0 = os.path.join(output_path,'txt','out0_normal.txt')
    out1=os.path.join(output_path,'txt','out1_crowd.txt')
    out2=os.path.join(output_path,'txt','out2_hlight.txt')
    out3=os.path.join(output_path,'txt','out3_shadow.txt')
    out4=os.path.join(output_path,'txt','out4_noline.txt')
    out5=os.path.join(output_path,'txt','out5_arrow.txt')
    out6=os.path.join(output_path,'txt','out6_curve.txt')
    out7=os.path.join(output_path,'txt','out7_cross.txt')
    out8=os.path.join(output_path,'txt','out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd']= read_helper(out1)
    res_all['res_night']= read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow']= read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve']= read_helper(out6)
    res_all['res_cross']= read_helper(out7)
    return res_all

def call_curvelane_eval(data_dir, exp_name,output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=5
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=224
    im_h=224
    x_factor = 224 / 2560
    y_factor = 224 / 1440
    frame=1
    list0 = os.path.join(data_dir, 'valid', 'valid_for_culane_style.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0=os.path.join(output_path,'txt','out0_curve.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    print('./evaluate -s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -x %s -y %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0, x_factor, y_factor))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s -x %s -y %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0, x_factor, y_factor))
    res_all = {}
    res_all['res_curve'] = read_helper(out0)
    return res_all