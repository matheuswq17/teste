import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk


SEED = 42


@dataclass
class SeriesMetadata:
    patient_id: str
    lesion_id: str
    modality: str
    series_path: Path
    mask_path: Path
    target: int


@dataclass
class LoadedImage:
    image: sitk.Image
    mask: sitk.Image
    metadata: SeriesMetadata


class SegmentationLoader:
    """Handle paired image and mask loading/validation for PROSTATEx."""

    def __init__(self, voxel_spacing: Tuple[float, float, float]):
        self.voxel_spacing = voxel_spacing

    @staticmethod
    def _read_dicom_series(series_dir: Path) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        reader.SetFileNames(dicom_names)
        return reader.Execute()

    @staticmethod
    def _read_segmentation(mask_path: Path) -> sitk.Image:
        if mask_path.suffix.lower() in {".nii", ".nii.gz"}:
            return sitk.ReadImage(str(mask_path))
        if mask_path.is_dir():
            return SegmentationLoader._read_dicom_series(mask_path)
        raise ValueError(f"Unsupported mask format: {mask_path}")

    @staticmethod
    def _validate_geometry(image: sitk.Image, mask: sitk.Image) -> None:
        if image.GetSize() != mask.GetSize():
            raise ValueError("Image and mask size mismatch; resampling or alignment required.")
        if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1e-3):
            raise ValueError("Image and mask spacing mismatch; please resample to common grid.")
        if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1e-3):
            raise ValueError("Image and mask origin mismatch; ensure same affine.")

    def load_case(self, metadata: SeriesMetadata) -> LoadedImage:
        image = self._read_dicom_series(metadata.series_path)
        mask = self._read_segmentation(metadata.mask_path)
        self._validate_geometry(image, mask)
        return LoadedImage(image=image, mask=mask, metadata=metadata)

    def resample_to_spacing(self, loaded: LoadedImage) -> LoadedImage:
        image = loaded.image
        mask = loaded.mask
        new_spacing = self.voxel_spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(image.GetSize(), image.GetSpacing(), new_spacing)]
        resampler.SetSize(new_size)

        resampled_image = resampler.Execute(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_mask = resampler.Execute(mask)
        self._validate_geometry(resampled_image, resampled_mask)
        return LoadedImage(image=resampled_image, mask=resampled_mask, metadata=loaded.metadata)

    @staticmethod
    def manifest_from_csv(manifest_path: Path) -> List[SeriesMetadata]:
        import pandas as pd

        df = pd.read_csv(manifest_path)
        required_cols = {"PatientID", "LesionID", "Modality", "ImagePath", "MaskPath", "Target"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        entries: List[SeriesMetadata] = []
        for _, row in df.iterrows():
            entries.append(
                SeriesMetadata(
                    patient_id=str(row["PatientID"]),
                    lesion_id=str(row["LesionID"]),
                    modality=str(row["Modality"]),
                    series_path=Path(row["ImagePath"]),
                    mask_path=Path(row["MaskPath"]),
                    target=int(row["Target"]),
                )
            )
        return entries

    @staticmethod
    def save_manifest(entries: List[SeriesMetadata], output_path: Path) -> None:
        manifest_json: List[Dict[str, str]] = []
        for e in entries:
            manifest_json.append(
                {
                    "PatientID": e.patient_id,
                    "LesionID": e.lesion_id,
                    "Modality": e.modality,
                    "ImagePath": str(e.series_path),
                    "MaskPath": str(e.mask_path),
                    "Target": e.target,
                }
            )
        output_path.write_text(json.dumps(manifest_json, indent=2))

