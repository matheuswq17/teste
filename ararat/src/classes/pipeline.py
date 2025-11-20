import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from .preprocessing import ImagePreprocessor
from .segmentation import LoadedImage, SegmentationLoader, SeriesMetadata


@dataclass
class RadiomicsExample:
    patient_id: str
    lesion_id: str
    modality: str
    target: int
    features: Dict[str, float]


class RadiomicsPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.output_root = Path(config["output_root"])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.export_dir = self.output_root / f"compete_no_overfit_FINAL_{self.timestamp}"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.loader = SegmentationLoader(tuple(config["voxel_spacing"]))
        self.preprocessor = ImagePreprocessor(tuple(config["voxel_spacing"]))
        self.extractor = featureextractor.RadiomicsFeatureExtractor(config["pyradiomics_params"])
        self.extractor.disableAllFeatures()
        for family in config.get("feature_families", []):
            self.extractor.enableFeatureClassByName(family.replace("original_", ""))
        self.extractor.enableAllImageTypes()

    def _extract_single(self, loaded: LoadedImage) -> Dict[str, float]:
        preprocessed = self.preprocessor.preprocess_loaded(loaded, normalize=True)
        result = self.extractor.execute(preprocessed.image, preprocessed.mask)
        clean_features = {k: float(v) for k, v in result.items() if k.startswith("original_")}
        return clean_features

    def build_dataset(self, manifest_path: Path) -> pd.DataFrame:
        entries = self.loader.manifest_from_csv(manifest_path)
        rows: List[RadiomicsExample] = []
        for entry in entries:
            loaded = self.loader.load_case(entry)
            feats = self._extract_single(loaded)
            rows.append(
                RadiomicsExample(
                    patient_id=entry.patient_id,
                    lesion_id=entry.lesion_id,
                    modality=entry.modality,
                    target=entry.target,
                    features=feats,
                )
            )
        records: List[Dict[str, float]] = []
        for r in rows:
            record = {
                "PatientID": r.patient_id,
                "LesionID": r.lesion_id,
                "Modality": r.modality,
                "Target": r.target,
            }
            record.update(r.features)
            records.append(record)
        df = pd.DataFrame(records)
        feature_csv = self.export_dir / "radiomics_features.csv"
        df.to_csv(feature_csv, index=False)
        manifest_copy = self.export_dir / "manifest_used.json"
        manifest_copy.write_text(Path(manifest_path).read_text())
        return df

    def save_config_snapshot(self) -> None:
        snapshot_path = self.export_dir / "config_snapshot.json"
        snapshot_path.write_text(json.dumps(self.config, indent=2))

