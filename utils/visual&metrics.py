import os
import pandas as pd
import numpy as np
import trimesh
from stl import mesh as ms

def point_to_plane_dist(A, B, C, D, x0, y0, z0, signed=True):
    n = np.array([A, B, C], dtype=float)
    val = A * x0 + B * y0 + C * z0 + D
    d = val / np.linalg.norm(n)
    return abs(d)

def plane_metrics(A1,B1,C1,D1, A2,B2,C2,D2,x0,y0,z0, deg=True):
    n1 = np.array([A1,B1,C1], dtype=float)
    n2 = np.array([A2,B2,C2], dtype=float)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    cos_th = np.clip(abs(np.dot(n1,n2)) / (n1n*n2n), 0.0, 1.0)
    theta = np.arccos(cos_th)
    if deg:
        theta = np.degrees(theta)

    d1 = point_to_plane_dist(A1, B1, C1, D1, x0, y0, z0)
    d2 = point_to_plane_dist(A2, B2, C2, D2, x0, y0, z0)

    diff = abs(d1 - d2)
    return theta, diff

def stl_numpy(stl_path):
    mesh1 = ms.Mesh.from_file(stl_path)
    locations = mesh1.vectors.reshape(-1, 3)
    return locations

model_path = '' # parent path of predicted plane
gt_path = '' # parent path of ground_truth plane
patient = '' # patient id

# model
msp_model_path = os.path.join(model_path, patient, 'pre_para.csv') # model predicted plane

landmarks_1_df = pd.read_csv(msp_model_path, header=None)

points_1 = landmarks_1_df.values  # shape: (5, 3)

A = points_1.tolist()[0][0]
B = points_1.tolist()[1][0]
C = points_1.tolist()[2][0]
D = points_1.tolist()[3][0]

# axis-aligned bounding box
bone_path = os.path.join(model_path,  patient, '3d_skull.stl') # 3d skull as axis-aligned bounding box to calculate error
bone_stl = stl_numpy(bone_path)

x_min = np.min(bone_stl[:, 0])
x_max = np.max(bone_stl[:, 0])
z_min = np.min(bone_stl[:, 2])
z_max = np.max(bone_stl[:, 2])
y_min = np.min(bone_stl[:, 1])
y_max = np.max(bone_stl[:, 1])

# gt
msp_gt_path = os.path.join(model_path, patient, 'gt_para.csv') # ground-truth plane

landmarks_2_df = pd.read_csv(msp_gt_path, header=None)

points_2 = landmarks_2_df.values

A_2 = points_2.tolist()[0][0]
B_2 = points_2.tolist()[1][0]
C_2 = points_2.tolist()[2][0]
D_2 = points_2.tolist()[3][0]

theta, diff_corner = plane_metrics(A, B, C, D, A_2, B_2, C_2, D_2,
                                   x_min, y_min, z_min)

_, diff_center = plane_metrics(A, B, C, D, A_2, B_2, C_2, D_2,
                               x_min, (y_min + y_max) / 2, (z_min + z_max) / 2)

diff = (diff_center + diff_corner) / 2

print("Angular error is {:.3f}°".format(theta))
print("Offset error (corner) is {:.3f} mm".format(diff_corner))
print("Offset error (center) is {:.3f} mm".format(diff_center))
print("Offset error is {:.3f} mm".format(diff))

resolution = 200
y_vals = np.linspace(y_min - 20, y_max + 20, resolution)
z_vals = np.linspace(z_min - 20, z_max + 20, resolution)
Y, Z = np.meshgrid(y_vals, z_vals)

X = (-B * Y - C * Z - D) / A
vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
faces = []
for i in range(resolution - 1):
    for j in range(resolution - 1):
        idx = i * resolution + j
        faces.append([idx, idx + 1, idx + resolution])
        faces.append([idx + 1, idx + resolution + 1, idx + resolution])
faces = np.array(faces)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

save_predict_plane_path = '' # visulization save path
mesh.export(save_predict_plane_path)