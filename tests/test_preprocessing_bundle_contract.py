import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.modules.setdefault("django.conf", type(sys)("django.conf"))
sys.modules["django.conf"].settings = type("Settings", (), {})()

from ml_pipeline.preprocessing_contract import (
    build_preprocessing_spec,
    build_versioned_evaluation_features,
    normalize_preprocessing_spec,
)
from services.predict_service import (
    BUNDLE_MODE_AWARE,
    BUNDLE_MODE_LEGACY,
    DiggPredictionService,
    RuntimeAssetBundle,
)


FEATURE_ORDER = [
    "duration_sec",
    "follower_count_log",
    "publish_hour",
    "avg_sentiment",
    "visual_brightness",
    "visual_saturation",
    "cut_frequency",
    "audio_bpm",
    "theme_encoded",
    "visual_impact",
    "sensory_pace",
    "sentiment_intensity",
    "audio_visual_energy",
    "content_density",
    "theme_ThemeA",
    "theme_Unknown",
]


class PassthroughScaler:
    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.mean_ = np.zeros(len(feature_names), dtype=float)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.loc[:, list(self.feature_names_in_)].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)


class EchoThemeEncodedModel:
    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        self.theme_encoded_idx = list(feature_names).index("theme_encoded")

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            values = X.to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
        return values[:, self.theme_encoded_idx]


def make_fitted_theme_encoding(theme_value, global_mean=1.5, smoothing_weight=10.0):
    return {
        "global_mean": float(global_mean),
        "smoothing_weight": float(smoothing_weight),
        "theme_count_by_theme": {"ThemeA": 12},
        "theme_local_mean_by_theme": {"ThemeA": 2.0},
        "theme_encoded_by_theme": {"ThemeA": float(theme_value)},
    }


def make_eval_df():
    return pd.DataFrame(
        [
            {
                "theme_label": "ThemeA",
                "duration_sec": 15.0,
                "follower_count_log": np.log1p(1200.0),
                "publish_hour": 18.0,
                "avg_sentiment": 0.55,
                "visual_brightness": 120.0,
                "visual_saturation": 90.0,
                "cut_frequency": 0.8,
                "audio_bpm": 110.0,
                "digg_count": 5000.0,
            }
        ]
    )


class TestPreprocessingBundleContract(unittest.TestCase):
    def setUp(self):
        DiggPredictionService._instance = None

    def tearDown(self):
        DiggPredictionService._instance = None

    def _build_service_without_disk_load(self):
        with patch.object(DiggPredictionService, "_load_assets", autospec=True, return_value=None):
            service = DiggPredictionService()
        service._activate_runtime_bundle(service._empty_runtime_bundle())
        return service

    def test_prep_round_trip_normalizes_training_schema(self):
        raw_spec = build_preprocessing_spec(
            version_id="v_roundtrip",
            feature_names_in=FEATURE_ORDER,
            known_theme_cols=["theme_ThemeA", "theme_Unknown"],
            fitted_theme_encoding=make_fitted_theme_encoding(theme_value=3.2),
            numeric_imputation_values={
                "avg_sentiment": 0.5,
                "visual_brightness": 128.0,
                "visual_saturation": 100.0,
                "cut_frequency": 0.5,
                "audio_bpm": 110.0,
            },
            follower_clip_upper=8.5,
        )

        normalized = normalize_preprocessing_spec(raw_spec, version_id="v_roundtrip")

        self.assertEqual(normalized["version_id"], "v_roundtrip")
        self.assertEqual(normalized["feature_names_in"], FEATURE_ORDER)
        self.assertEqual(normalized["preprocessing"]["feature_order"], FEATURE_ORDER)
        self.assertEqual(
            normalized["preprocessing"]["known_theme_cols"],
            ["theme_ThemeA", "theme_Unknown"],
        )
        self.assertAlmostEqual(
            normalized["theme_encoding"]["theme_stats"]["ThemeA"]["theme_encoded"],
            3.2,
        )
        self.assertAlmostEqual(normalized["bayesian_global_mean"], 1.5)
        self.assertAlmostEqual(
            normalized["numeric_imputations"]["visual_brightness"],
            128.0,
        )
        self.assertAlmostEqual(
            normalized["clip_values"]["follower_count_log"]["upper"],
            8.5,
        )

    def test_bundle_aware_loader_uses_manifest_linked_prep_for_inference(self):
        prep_spec = build_preprocessing_spec(
            version_id="v_bundle",
            feature_names_in=FEATURE_ORDER,
            known_theme_cols=["theme_ThemeA", "theme_Unknown"],
            fitted_theme_encoding=make_fitted_theme_encoding(theme_value=3.4),
            numeric_imputation_values={
                "avg_sentiment": 0.5,
                "visual_brightness": 128.0,
                "visual_saturation": 100.0,
                "cut_frequency": 0.5,
                "audio_bpm": 110.0,
            },
            follower_clip_upper=9.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            joblib.dump(EchoThemeEncodedModel(FEATURE_ORDER), artifacts_dir / "linked_model.pkl")
            joblib.dump(PassthroughScaler(FEATURE_ORDER), artifacts_dir / "linked_scaler.pkl")
            with open(artifacts_dir / "linked_prep_bundle.json", "w", encoding="utf-8") as f:
                json.dump(prep_spec, f, ensure_ascii=False, indent=2)

            manifest = {
                "current_version": "v_bundle",
                "previous_version": None,
                "bayesian_global_mean": 1.5,
                "bundle_artifacts": {
                    "version_id": "v_bundle",
                    "model_file": "linked_model.pkl",
                    "scaler_file": "linked_scaler.pkl",
                    "prep_spec_file": "linked_prep_bundle.json",
                    "bundle_complete": True,
                },
                "prep_spec_file": "linked_prep_bundle.json",
                "feature_names_in": FEATURE_ORDER,
            }
            with open(artifacts_dir / "version_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            service = self._build_service_without_disk_load()
            service.artifacts_dir = artifacts_dir
            service.manifest_path = artifacts_dir / "version_manifest.json"
            service._load_assets()

            self.assertEqual(service.bundle_mode, BUNDLE_MODE_AWARE)
            features = service._build_model_input_features(
                {
                    "theme_label": "ThemeA",
                    "duration_sec": 15.0,
                    "follower_count": 1200.0,
                    "publish_hour": 18,
                    "avg_sentiment": 0.55,
                    "visual_brightness": 120.0,
                    "visual_saturation": 90.0,
                    "cut_frequency": 0.8,
                    "audio_bpm": 110.0,
                },
                display_theme_baseline={"mean": 999999.0, "count": 9999},
            )

            self.assertAlmostEqual(float(features.at[0, "theme_encoded"]), 3.4)
            raw_pred = service._predict_raw_model_log(features, runtime_bundle=service._get_active_bundle())
            self.assertAlmostEqual(float(raw_pred), 3.4)

    def test_legacy_path_does_not_use_display_baseline_as_model_input_truth(self):
        service = self._build_service_without_disk_load()
        service._activate_runtime_bundle(
            RuntimeAssetBundle(
                version_id="legacy_v1",
                model=EchoThemeEncodedModel(FEATURE_ORDER),
                scaler=PassthroughScaler(FEATURE_ORDER),
                prep=None,
                manifest={"bayesian_global_mean": 1.75},
                ordered_feature_names=tuple(FEATURE_ORDER),
                bundle_mode=BUNDLE_MODE_LEGACY,
                feature_topology_source="manifest.feature_names_in",
            )
        )

        features = service._build_model_input_features(
            {
                "theme_label": "ThemeA",
                "duration_sec": 12.0,
                "follower_count": 800.0,
                "publish_hour": 20,
            },
            display_theme_baseline={"mean": 250000.0, "count": 5000},
        )

        self.assertAlmostEqual(float(features.at[0, "theme_encoded"]), 1.75)
        self.assertNotAlmostEqual(float(features.at[0, "theme_encoded"]), float(np.log1p(250000.0)))

    def test_showdown_feature_rebuild_is_version_specific_for_each_bundle(self):
        eval_df = make_eval_df()
        scaler = PassthroughScaler(FEATURE_ORDER)

        challenger_spec = build_preprocessing_spec(
            version_id="v_challenger",
            feature_names_in=FEATURE_ORDER,
            known_theme_cols=["theme_ThemeA", "theme_Unknown"],
            fitted_theme_encoding=make_fitted_theme_encoding(theme_value=4.2),
            numeric_imputation_values={
                "avg_sentiment": 0.5,
                "visual_brightness": 128.0,
                "visual_saturation": 100.0,
                "cut_frequency": 0.5,
                "audio_bpm": 110.0,
            },
            follower_clip_upper=9.0,
        )
        champion_spec = build_preprocessing_spec(
            version_id="v_champion",
            feature_names_in=FEATURE_ORDER,
            known_theme_cols=["theme_ThemeA", "theme_Unknown"],
            fitted_theme_encoding=make_fitted_theme_encoding(theme_value=2.1),
            numeric_imputation_values={
                "avg_sentiment": 0.5,
                "visual_brightness": 128.0,
                "visual_saturation": 100.0,
                "cut_frequency": 0.5,
                "audio_bpm": 110.0,
            },
            follower_clip_upper=9.0,
        )

        challenger_eval = build_versioned_evaluation_features(
            eval_df=eval_df,
            expected_features=FEATURE_ORDER,
            scaler=scaler,
            prep_spec=challenger_spec,
            manifest_payload={"bayesian_global_mean": 1.5},
        )
        champion_eval = build_versioned_evaluation_features(
            eval_df=eval_df,
            expected_features=FEATURE_ORDER,
            scaler=scaler,
            prep_spec=champion_spec,
            manifest_payload={"bayesian_global_mean": 1.5},
        )

        self.assertAlmostEqual(float(challenger_eval.at[0, "theme_encoded"]), 4.2)
        self.assertAlmostEqual(float(champion_eval.at[0, "theme_encoded"]), 2.1)
        self.assertNotEqual(
            float(challenger_eval.at[0, "theme_encoded"]),
            float(champion_eval.at[0, "theme_encoded"]),
        )


if __name__ == "__main__":
    unittest.main()
