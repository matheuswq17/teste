from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .segmentation import LoadedImage


@dataclass
class PreprocessedImage:
    image: sitk.Image
    mask: sitk.Image
    metadata: Dict[str, str]


class ImagePreprocessor:
    def __init__(self, voxel_spacing: Tuple[float, float, float]):
        self.voxel_spacing = voxel_spacing

    def zscore_normalize(self, image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        mu = np.mean(array)
        sigma = np.std(array) + 1e-8
        normalized = (array - mu) / sigma
        out = sitk.GetImageFromArray(normalized)
        out.CopyInformation(image)
        return out

    def preprocess_loaded(self, loaded: LoadedImage, normalize: bool = True) -> PreprocessedImage:
        from .segmentation import SegmentationLoader

        resampled = SegmentationLoader(self.voxel_spacing).resample_to_spacing(loaded)
        image = resampled.image
        if normalize:
            image = self.zscore_normalize(image)
        return PreprocessedImage(image=image, mask=resampled.mask, metadata=resampled.metadata.__dict__)


class TabularPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> None:
        self.imputer.fit(X)
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        self.scaler.fit(X_imputed)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imputed = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        scaled = pd.DataFrame(self.scaler.transform(X_imputed), columns=X.columns)
        return scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump({"imputer": self.imputer, "scaler": self.scaler}, path)

    def load(self, path: Path) -> None:
        import joblib

        bundle = joblib.load(path)
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]

