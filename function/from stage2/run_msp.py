import sys
import os
import numpy as np
import vtk
import torch
from sklearn.covariance import EllipticEnvelope
from scipy.linalg import lstsq
import scipy.cluster.hierarchy as hierarchy
from numpy.linalg import svd
from scipy.spatial import distance_matrix
from stl import mesh
sys.path.append('/function/from_stage2') # setting root path
from pathlib import Path


def stl_numpy(stl_path):
    mesh1 = mesh.Mesh.from_file(stl_path)
    locations = mesh1.vectors.reshape(-1, 3)
    return locations

def gen_plane(point_0, point_1, i):
    value_c = point_0.reshape((-1, 1, 3)) - point_1.reshape((1, -1, 3))
    select_indexs_1 = point_0[np.where((np.abs(value_c).sum(axis=2) < i).sum(axis=1))]
    select_indexs_2 = point_1[np.where((np.abs(value_c).sum(axis=2) < i).sum(axis=0))]
    selet_data = np.vstack([select_indexs_1, select_indexs_2])
    detector = EllipticEnvelope()
    detector.fit(selet_data)
    selet_index = detector.predict(selet_data)
    p1 = selet_data[np.where(selet_index == 1)]
    X = p1[:, 0]
    Y = p1[:, 1]
    Z = p1[:, 2]
    A = np.c_[X, Y, np.ones(X.shape)]
    C, resid, rank, s = lstsq(A, Z)
    a, b, c = C
    return a, b, c

def find_plane_point(A, B, D, thread_num):

    threshold = thread_num
    i, j = np.where(D <= threshold)
    Z = hierarchy.linkage(np.c_[i, j], method='single', metric='euclidean')
    C = hierarchy.fcluster(Z, t=0.6, criterion='distance')
    plane_point = []
    for k in range(1, C.max() + 1):
        idx = np.where(C == k)[0]
        center_point = (A[i[idx]][0] + B[j[idx]][0]) * 0.5
        center_point = center_point.astype(int)
        plane_point.extend(center_point.tolist())
    plane_point = np.array(plane_point).reshape(-1, 3)
    return plane_point

def best_fit_plane(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    H = np.dot(centered_points.T, centered_points)
    U, S, V = svd(H)
    normal = V[2, :]
    D = -np.dot(normal, centroid)
    return normal, D

def generate_plane_point_cloud_from_yz(normal, D, size=1000, yrange=(-10, 10), zrange=(-10, 10)):
    y = np.random.uniform(yrange[0], yrange[1], size)
    z = np.random.uniform(zrange[0], zrange[1], size)
    x = (-D - normal[1] * y - normal[2] * z) / normal[0]
    plane_points = np.vstack((x, y, z)).T
    return plane_points

def point_to_plane_distance(point, normal, D):
    x0, y0, z0 = point
    nx, ny, nz = normal
    numerator = abs(nx * x0 + ny * y0 + nz * z0 + D)
    denominator = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    distance = numerator / denominator
    return distance

def save_stl_variants(
    in_stl,
    out_hi_stl,
    out_small_stl,
    smooth_iterations=30,
    smooth_relaxation=0.5,
    target_reduction=0.80,
    max_triangles=None,
    boundary_smoothing=True
):
    in_stl = str(in_stl); out_hi_stl = str(out_hi_stl); out_small_stl = str(out_small_stl)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(in_stl)
    reader.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(reader.GetOutputPort())
    tri.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.PointMergingOn()
    clean.Update()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(clean.GetOutputPort())
    smooth.SetNumberOfIterations(smooth_iterations)
    smooth.SetRelaxationFactor(smooth_relaxation)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn() if boundary_smoothing else smooth.BoundarySmoothingOff()
    smooth.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smooth.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    w_hi = vtk.vtkSTLWriter()
    w_hi.SetInputConnection(normals.GetOutputPort())
    w_hi.SetFileName(out_hi_stl)
    w_hi.SetFileTypeToBinary()
    w_hi.Write()

    deci = vtk.vtkQuadricDecimation()
    deci.SetInputConnection(normals.GetOutputPort())

    current_tris = normals.GetOutput().GetNumberOfCells()
    if max_triangles is not None and current_tris > 0:
        keep_ratio = min(1.0, max(0.0, max_triangles / float(current_tris)))
        target_reduction = 1.0 - keep_ratio

    target_reduction = float(max(0.0, min(target_reduction, 0.98)))
    deci.SetTargetReduction(target_reduction)

    deci.Update()

    clean2 = vtk.vtkCleanPolyData()
    clean2.SetInputConnection(deci.GetOutputPort())
    clean2.Update()

    w_small = vtk.vtkSTLWriter()
    w_small.SetInputConnection(clean2.GetOutputPort())
    w_small.SetFileName(out_small_stl)
    w_small.SetFileTypeToBinary()
    w_small.Write()

    return Path(out_hi_stl), Path(out_small_stl)

patient = '' # patient_name
input_stl = '' # stl file after AGR
out_dir = '' # output directory


print(patient, '--------------start predicting MSP--------------')

'''trans2point-cloud'''
upper_path = input_stl
upper_stl = stl_numpy(upper_path)

upper_len = len(upper_stl)
n_upper = upper_len // 90000
upper_points = upper_stl[::n_upper]

rgb = np.full_like(upper_points, 128, dtype=int)
final_data = np.hstack((upper_points, rgb))

save_path = os.path.join(out_dir, 'upper_points_processed.txt')
np.savetxt(save_path, final_data)

'''Gen MSP'''
sys.path.remove('/function/from_stage2')  # removing root path
sys.path.append('/OpenPoints_framework') # resetting root path

from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port, load_checkpoint_inv
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.models import build_model_from_cfg
from openpoints.transforms import build_transforms_from_cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # setting gpu id

cfg = EasyConfig()
cfg.load('/OpenPoints_framework/cfgs/cmf_msp/deepgcn.yaml', recursive=True) # deepgcn.yaml
cfg.weight = '' # weight
cfg.num_points = 4096

if cfg.seed is None:
    cfg.seed = np.random.randint(1, 10000)

cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
cfg.sync_bn = cfg.world_size > 1

cfg.mp = False
cfg.distributed = False
cfg.sync_bn = False

cfg.feature_keys = 'x,heights'

if cfg.model.get('in_channels', None) is None:
    cfg.model.in_channels = cfg.model.encoder_args.in_channels
model = build_model_from_cfg(cfg.model).to(cfg.rank)

load_checkpoint(model, pretrained_path=cfg.weight)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

txt_path = save_path
data = np.loadtxt(txt_path)
ori_coord = data[:, :3].astype(np.float32)
ori_feat = data[:, 3:6].astype(np.float32) / 255.

trans = ori_coord.min(0)
ori_coord -= ori_coord.min(0)

N = ori_coord.shape[0]
if N >= cfg.num_points:
    choice = np.random.choice(N, cfg.num_points, replace=False)
else:
    choice = np.random.choice(N, cfg.num_points, replace=True)

coord = ori_coord[choice]
feat = ori_feat[choice]

x_feat = coord

data_dict = {
    'x': x_feat,
    'pos': coord
}

data_transform = build_transforms_from_cfg('val', cfg.datatransforms)
data_dict = data_transform(data_dict)
heights = coord[:, 2:3].astype(np.float32)
data_dict['heights'] = torch.from_numpy(heights)

for key in data_dict:
    if isinstance(data_dict[key], torch.Tensor):
        data_dict[key] = data_dict[key].unsqueeze(0)

for k in data_dict:
    data_dict[k] = data_dict[k].to(device)

keys = data_dict.keys() if callable(data_dict.keys) else data_dict.keys
data_dict['x'] = get_features_by_keys(data_dict, cfg.feature_keys)
vote_pool = torch.zeros(1, 2, cfg.num_points).cuda()
data_dict['x'] = data_dict['x'].contiguous() # pointnet++
for _ in range(3):
    logits = model(data_dict)
    vote_pool += logits
seg_logits = vote_pool/3
preds = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()

coord += trans

cur_pred_val = preds.reshape(-1, 1)
predict = np.hstack((coord, cur_pred_val))
save_folder = os.path.join(out_dir)
save_path = os.path.join(save_folder, 'predict.txt')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
np.savetxt(save_path, predict)

bool_array = cur_pred_val.astype(bool).flatten()
point_0 = coord[~bool_array]
point_1 = coord[bool_array]

A_p = point_0
B_p = point_1
T_p = distance_matrix(A_p, B_p)
thread_num = 10
plane_point = find_plane_point(A_p, B_p, T_p, thread_num)
normal_p, D_p = best_fit_plane(plane_point)

A_pre = normal_p[0]
B_pre = normal_p[1]
C_pre = normal_p[2]
D_pre = D_p

pre_para = [A_pre, B_pre, C_pre, D_pre]
save_para = os.path.join(save_folder, 'pre_para.csv')
np.savetxt(save_para, pre_para, delimiter=",", comments="", fmt="%s")