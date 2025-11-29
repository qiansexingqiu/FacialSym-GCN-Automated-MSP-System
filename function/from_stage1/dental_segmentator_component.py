import os
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union, Dict
import math
import numpy as np
import nibabel as nib

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

from skimage import measure
import trimesh

from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data

DEFAULT_LABEL_MAP: Dict[int, str] = {
    1: "upper_skull",
    2: "mandible",
    3: "upper_teeth",
    4: "lower_teeth",
    5: "mandibular_canal",
}

DEFAULT_NAME2ID = {v: k for k, v in DEFAULT_LABEL_MAP.items()}

def _dcm_or_nii_to_single_nii(input_path: Union[str, Path], out_nii: Path, cut_mm) -> None:
    input_path = Path(input_path)
    out_nii.parent.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        if sitk is None:
            raise RuntimeError("SimpleITK 未安装，无法从 DICOM 读取。请 pip install SimpleITK")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(input_path))
        if not series_ids:
            raise ValueError(f"未在 {input_path} 发现 DICOM 序列")

        best_files = None
        max_len = -1
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(input_path), sid)
            if len(files) > max_len:
                max_len = len(files)
                best_files = files

        reader.SetFileNames(best_files)
        img = reader.Execute()


        size = list(img.GetSize())
        spacing = img.GetSpacing()

        p0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
        p1 = img.TransformIndexToPhysicalPoint((0, 0, size[2] - 1))

        inferior_is_high_index = (p1[2] < p0[2])

        remove_nz = math.ceil(cut_mm / spacing[2])
        remove_nz = min(remove_nz, size[2] - 1)

        if inferior_is_high_index:
            z_start = 0
            z_end = size[2] - remove_nz
        else:
            z_start = remove_nz
            z_end = size[2]

        start = [0, 0, z_start]
        new_size = [size[0], size[1], z_end - z_start]

        cropped = sitk.RegionOfInterest(img, new_size, start)

        sitk.WriteImage(cropped, str(out_nii), True)
    else:
        suffix = "".join(input_path.suffixes)
        if suffix not in [".nii", ".nii.gz"]:
            raise ValueError(f"不支持的文件格式：{input_path.name}（仅支持 DICOM 目录或 .nii/.nii.gz）")
        shutil.copy2(str(input_path), str(out_nii))


def _export_label_to_stl(seg_nii: Union[str, Path],
                         out_stl: Union[str, Path],
                         label_id: int,
                         smoothing_lambda: float = 0.0,
                         smoothing_iterations: int = 0) -> Optional[Path]:
    seg_nii = Path(seg_nii)
    out_stl = Path(out_stl)
    out_stl.parent.mkdir(parents=True, exist_ok=True)

    seg = nib.load(str(seg_nii))
    data = seg.get_fdata().astype(np.int16)
    spacing = seg.header.get_zooms()[:3]

    if isinstance(label_id, (list, tuple, set)):
        mask = np.isin(data, list(label_id))
    else:
        mask = (data == label_id)

    if mask.sum() == 0:
        return None

    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8), level=0.5, spacing=spacing
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if smoothing_iterations > 0 and smoothing_lambda > 0:
        mesh = mesh.smoothed(filter='laplacian',
                             lamb=smoothing_lambda,
                             iterations=smoothing_iterations)

    mesh.export(str(out_stl))
    return out_stl


class DentalSegComponent:

    def __init__(self,
                 dataset_name: str = "Dataset112_DentalSegmentator_v100",
                 configuration: str = "3d_fullres",
                 folds: Union[str, list] = "0",
                 nnunet_results: Optional[Union[str, Path]] = None,
                 label_map: Optional[Dict[int, str]] = None):
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.folds = folds
        self.label_map = label_map or DEFAULT_LABEL_MAP

        self.nnunet_results = Path(nnunet_results) if nnunet_results else Path(os.environ.get("nnUNet_results", ""))
        if not self.nnunet_results.exists():
            raise EnvironmentError("未找到 nnUNet_results 目录。请设置环境变量 nnUNet_results 或在构造函数中传入。")

        ds_dir = self.nnunet_results / self.dataset_name
        if not ds_dir.exists():
            raise FileNotFoundError(f"未在 {self.nnunet_results} 下找到 {self.dataset_name}。请确认权重解压位置。")

    def _folds_as_list(self):
        if isinstance(self.folds, str):
            return self.folds.split()
        return [str(f) for f in self.folds]

    def run_single(self,
                   input_path: Union[str, Path],
                   out_dir: Union[str, Path],
                   label: Union[int, str] = "mandible",
                   case_id: Optional[str] = None,
                   smoothing_lambda: float = 0.0,
                   smoothing_iterations: int = 0,
                   cut_mm: int = 0,
                   keep_pred_nii: bool = False) -> Path:

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(label, str):
            name2id = {v: k for k, v in self.label_map.items()}
            if label not in name2id:
                raise KeyError(f"未知标签名：{label}，可选：{list(name2id.keys())}")
            label_id = name2id[label]
            label_name = label
        else:
            label_id = int(label)
            label_name = self.label_map.get(label_id, f"label_{label_id}")

        with tempfile.TemporaryDirectory(prefix="dentseg_") as workdir:
            workdir = Path(workdir)
            imagesTs = workdir / "imagesTs"
            preds = workdir / "preds"
            imagesTs.mkdir(exist_ok=True)

            if case_id is None:
                case_id = Path(input_path).stem[:-4]
            in_nii = imagesTs / f"{case_id}_0000.nii.gz"
            cut_mm = cut_mm
            _dcm_or_nii_to_single_nii(input_path, in_nii, cut_mm)

            env = os.environ.copy()
            env["nnUNet_results"] = str(self.nnunet_results)

            folds = tuple(int(f) for f in self._folds_as_list())

            predict_from_raw_data(
                str(imagesTs),  # input_folder
                str(preds),  # output_folder
                str(self.nnunet_results / self.dataset_name / f"nnUNetTrainer__nnUNetPlans__{self.configuration}"),
                # model_folder
                folds,  # (0,)
                0.5,  # tile_step_size
                True,  # use_gaussian
                True,  # use_mirroring
                True,  # perform_everything_on_gpu
                True,  # verbose
                False,  # save_probabilities
                True,  # overwrite
                "checkpoint_final.pth",  # checkpoint_name
                1,  # num_processes_preprocessing
                1  # num_processes_segmentation_export
            )

            seg_pred = preds / f"{case_id}.nii.gz"
            if not seg_pred.exists():

                cands = list(preds.glob("*.nii.gz"))
                if len(cands) != 1:
                    raise FileNotFoundError(f"未找到预测结果 NIfTI，preds 下有：{[p.name for p in cands]}")
                seg_pred = cands[0]


            out_stl = out_dir / f"{case_id}__{label_name}.stl"
            stl_path = _export_label_to_stl(
                seg_pred, out_stl, [1, 3],
                smoothing_lambda=smoothing_lambda,
                smoothing_iterations=smoothing_iterations
            )

            all_stl = out_dir / "bone.stl"
            stl_path = _export_label_to_stl(
                seg_pred, all_stl, [1, 2, 3, 4, 5],
                smoothing_lambda=smoothing_lambda,
                smoothing_iterations=smoothing_iterations
            )


            if stl_path is None:
                raise ValueError(f"该病例中目标标签({label_id}={label_name})为空。")

            return stl_path