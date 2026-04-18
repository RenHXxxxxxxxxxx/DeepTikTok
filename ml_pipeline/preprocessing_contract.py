import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PREPROCESSING_SCHEMA_VERSION = "version_owned_preprocessing_v2"
DEFAULT_THEME_GLOBAL_MEAN_LOG = float(np.log1p(10000.0))


def build_feature_matrix(df, fitted_theme_encoding, known_theme_cols):
    """Build the model feature frame from version-owned theme priors."""
    df_features = df.copy()
    df_features["theme_encoded"] = (
        df_features["theme_label"]
        .map(fitted_theme_encoding["theme_encoded_by_theme"])
        .fillna(fitted_theme_encoding["global_mean"])
        .astype(float)
    )

    theme_dummies = pd.get_dummies(df_features["theme_label"], prefix="theme")
    theme_dummies = theme_dummies.reindex(columns=known_theme_cols, fill_value=0)

    base_features = [
        "duration_sec",
        "follower_count_log",
        "publish_hour",
        "avg_sentiment",
        "visual_brightness",
        "visual_saturation",
        "cut_frequency",
        "audio_bpm",
        "theme_encoded",
    ]
    return pd.concat([df_features[base_features], theme_dummies], axis=1)


def add_derived_features(df):
    """Add derived model features after base numeric/theme preprocessing."""
    df_features = df.copy()
    df_features.loc[:, "visual_impact"] = (
        df_features["visual_brightness"] * df_features["visual_saturation"]
    ) / 1000.0
    df_features.loc[:, "sensory_pace"] = df_features["audio_bpm"] * df_features["cut_frequency"]
    df_features.loc[:, "sentiment_intensity"] = abs(df_features["avg_sentiment"] - 0.5) * 2
    df_features.loc[:, "audio_visual_energy"] = (
        df_features["visual_brightness"] * df_features["audio_bpm"]
    ) / 1000.0
    df_features.loc[:, "content_density"] = (
        df_features["cut_frequency"] / (df_features["duration_sec"] + 1)
    )
    return df_features


def apply_numeric_preprocessing(df, follower_clip_upper, numeric_imputation_values):
    """Apply fitted numeric clipping/imputation without re-fitting on eval data."""
    df_processed = df.copy()
    df_processed.loc[:, "follower_count_log"] = df_processed["follower_count_log"].clip(
        upper=follower_clip_upper
    )
    for col, impute_value in numeric_imputation_values.items():
        df_processed.loc[:, col] = df_processed[col].fillna(impute_value)
    return df_processed


def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value) or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _get_nested_value(data, path):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _pick_first(data, *paths):
    for path in paths:
        value = _get_nested_value(data, path)
        if value is not None:
            return value
    return None


def _pick_float_from_mapping(mapping, *keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key not in mapping:
            continue
        numeric = _safe_float(mapping.get(key), default=None)
        if numeric is not None:
            return float(numeric)
    return None


def _coerce_numeric_map(raw_mapping):
    numeric_map = {}
    if not isinstance(raw_mapping, dict):
        return numeric_map

    for feature_name, raw_value in raw_mapping.items():
        if isinstance(raw_value, dict):
            numeric_value = _pick_float_from_mapping(
                raw_value, "value", "median", "default", "fill_value", "impute_value"
            )
        else:
            numeric_value = _safe_float(raw_value, default=None)

        if numeric_value is not None:
            numeric_map[str(feature_name)] = float(numeric_value)

    return numeric_map


def _coerce_clip_map(raw_mapping):
    clip_map = {}
    if not isinstance(raw_mapping, dict):
        return clip_map

    for feature_name, raw_value in raw_mapping.items():
        lower = None
        upper = None
        if isinstance(raw_value, dict):
            lower = _pick_float_from_mapping(raw_value, "lower", "min")
            upper = _pick_float_from_mapping(raw_value, "upper", "max", "value")
        else:
            upper = _safe_float(raw_value, default=None)

        if lower is not None or upper is not None:
            clip_map[str(feature_name)] = {"lower": lower, "upper": upper}

    return clip_map


def _coerce_theme_stats(raw_theme_stats, global_mean_log, smoothing_weight):
    theme_stats = {}
    if isinstance(raw_theme_stats, dict):
        items = raw_theme_stats.items()
    elif isinstance(raw_theme_stats, list):
        items = []
        for item in raw_theme_stats:
            if not isinstance(item, dict):
                continue
            theme_label = str(item.get("theme_label") or item.get("theme") or item.get("label") or "")
            if theme_label:
                items.append((theme_label, item))
    else:
        items = []

    for theme_name, raw_entry in items:
        entry = {}
        if isinstance(raw_entry, dict):
            raw_count = _pick_float_from_mapping(raw_entry, "count", "sample_count", "n")
            if raw_count is not None:
                entry["count"] = int(max(raw_count, 0.0))

            local_mean_log = _pick_float_from_mapping(
                raw_entry, "local_mean_log", "local_mean", "mean_log"
            )
            if local_mean_log is None and "mean" in raw_entry:
                raw_mean = _safe_float(raw_entry.get("mean"), default=None)
                if raw_mean is not None:
                    local_mean_log = float(np.log1p(raw_mean))
            if local_mean_log is not None:
                entry["local_mean_log"] = float(local_mean_log)

            theme_encoded = _pick_float_from_mapping(
                raw_entry, "theme_encoded", "encoded", "encoded_value"
            )
            if theme_encoded is not None:
                entry["theme_encoded"] = float(theme_encoded)

        else:
            numeric_value = _safe_float(raw_entry, default=None)
            if numeric_value is not None:
                entry["theme_encoded"] = float(numeric_value)

        if "theme_encoded" not in entry:
            if "local_mean_log" in entry and global_mean_log is not None:
                count = float(entry.get("count", 0))
                denominator = count + float(smoothing_weight or 0.0)
                entry["theme_encoded"] = (
                    (count * entry["local_mean_log"] + float(smoothing_weight or 0.0) * global_mean_log)
                    / denominator
                    if denominator > 0
                    else float(global_mean_log)
                )
            elif global_mean_log is not None:
                entry["theme_encoded"] = float(global_mean_log)

        if entry:
            theme_stats[str(theme_name)] = entry

    return theme_stats


def build_preprocessing_spec(
    version_id,
    feature_names_in,
    known_theme_cols,
    fitted_theme_encoding,
    numeric_imputation_values,
    follower_clip_upper,
    producer="ml_pipeline.train_master_arena",
):
    """Build the canonical JSON-serializable version-owned preprocessing spec."""
    theme_stats = {}
    for theme_name, count in fitted_theme_encoding["theme_count_by_theme"].items():
        theme_stats[str(theme_name)] = {
            "count": int(count),
            "local_mean_log": float(fitted_theme_encoding["theme_local_mean_by_theme"][theme_name]),
            "theme_encoded": float(fitted_theme_encoding["theme_encoded_by_theme"][theme_name]),
        }

    feature_order = [str(col) for col in feature_names_in]
    known_theme_cols = [str(col) for col in known_theme_cols]
    bayesian_global_mean = float(fitted_theme_encoding["global_mean"])
    smoothing_weight = float(fitted_theme_encoding["smoothing_weight"])
    numeric_imputations = {
        str(col): float(value) for col, value in numeric_imputation_values.items()
    }
    clip_values = {
        "follower_count_log": {
            "upper": float(follower_clip_upper),
        }
    }
    theme_encoding = {
        "encoding_type": "smoothed_theme_target_mean_log1p",
        "target_space": "log1p_digg_count",
        "unseen_theme_fallback": "bayesian_global_mean",
        "global_mean_log": bayesian_global_mean,
        "smoothing_weight": smoothing_weight,
        "theme_stats": theme_stats,
        "theme_statistics": theme_stats,
    }

    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "derivation_version": "train_master_arena_prep_v2",
        "version_id": str(version_id),
        "created_at": datetime.datetime.now().isoformat(),
        "producer": producer,
        "expected_consumer_mode": "version_owned_preprocessing_metadata",
        "feature_names_in": feature_order,
        "known_theme_cols": known_theme_cols,
        "bayesian_global_mean": bayesian_global_mean,
        "theme_encoding": theme_encoding,
        "numeric_imputations": numeric_imputations,
        "clip_values": clip_values,
        "preprocessing": {
            "feature_names_in": feature_order,
            "feature_order": feature_order,
            "known_theme_cols": known_theme_cols,
            "theme_encoding": theme_encoding,
            "numeric_imputations": numeric_imputations,
            "numeric_defaults": numeric_imputations,
            "clip_values": clip_values,
            "numeric_preprocessing": {
                "numeric_imputation_values": numeric_imputations,
                "follower_count_log_clip_upper": float(follower_clip_upper),
            },
        },
    }


def normalize_preprocessing_spec(raw_spec, version_id=None):
    """Normalize legacy and current spec dialects into the canonical schema."""
    if not isinstance(raw_spec, dict):
        return None

    source_version = str(
        raw_spec.get("version_id")
        or version_id
        or ""
    )

    feature_order = _pick_first(
        raw_spec,
        ("feature_names_in",),
        ("feature_order",),
        ("preprocessing", "feature_names_in"),
        ("preprocessing", "feature_order"),
        ("model_input", "feature_names_in"),
        ("model_input", "feature_order"),
    )
    if isinstance(feature_order, tuple):
        feature_order = list(feature_order)
    if not isinstance(feature_order, list):
        feature_order = []
    feature_order = [str(name) for name in feature_order]

    known_theme_cols = _pick_first(
        raw_spec,
        ("known_theme_cols",),
        ("preprocessing", "known_theme_cols"),
        ("theme_columns",),
    )
    if isinstance(known_theme_cols, tuple):
        known_theme_cols = list(known_theme_cols)
    if not isinstance(known_theme_cols, list):
        known_theme_cols = [name for name in feature_order if str(name).startswith("theme_")]
    known_theme_cols = [str(name) for name in known_theme_cols]

    bayesian_global_mean = _pick_first(
        raw_spec,
        ("bayesian_global_mean",),
        ("global_prior",),
        ("theme_encoding", "global_mean_log"),
        ("theme_encoding", "global_mean"),
        ("theme_encoding", "global_prior"),
        ("preprocessing", "bayesian_global_mean"),
        ("preprocessing", "global_prior"),
        ("preprocessing", "theme_encoding", "global_mean_log"),
        ("preprocessing", "theme_encoding", "global_mean"),
        ("preprocessing", "theme_encoding", "global_prior"),
    )
    bayesian_global_mean = _safe_float(
        bayesian_global_mean,
        default=DEFAULT_THEME_GLOBAL_MEAN_LOG,
    )

    smoothing_weight = _pick_first(
        raw_spec,
        ("theme_encoding", "smoothing_weight"),
        ("theme_encoding", "weight"),
        ("theme_encoding", "bayesian_weight"),
        ("bayesian_weight",),
        ("preprocessing", "theme_encoding", "smoothing_weight"),
        ("preprocessing", "theme_encoding", "weight"),
        ("preprocessing", "theme_encoding", "bayesian_weight"),
    )
    smoothing_weight = _safe_float(smoothing_weight, default=10.0)

    raw_theme_stats = _pick_first(
        raw_spec,
        ("theme_stats",),
        ("theme_priors",),
        ("theme_encoding", "theme_stats"),
        ("theme_encoding", "theme_statistics"),
        ("theme_encoding", "per_theme"),
        ("preprocessing", "theme_stats"),
        ("preprocessing", "theme_encoding", "theme_stats"),
        ("preprocessing", "theme_encoding", "theme_statistics"),
        ("preprocessing", "theme_encoding", "per_theme"),
    )
    theme_stats = _coerce_theme_stats(raw_theme_stats, bayesian_global_mean, smoothing_weight)

    numeric_imputations = _coerce_numeric_map(
        _pick_first(
            raw_spec,
            ("numeric_imputations",),
            ("numeric_defaults",),
            ("imputations",),
            ("numeric_preprocessing", "numeric_imputation_values"),
            ("preprocessing", "numeric_imputations"),
            ("preprocessing", "numeric_defaults"),
            ("preprocessing", "imputations"),
            ("preprocessing", "numeric_preprocessing", "numeric_imputation_values"),
        )
    )

    clip_values = _coerce_clip_map(
        _pick_first(
            raw_spec,
            ("clip_values",),
            ("feature_clips",),
            ("preprocessing", "clip_values"),
            ("preprocessing", "feature_clips"),
        )
    )
    follower_clip_upper = _pick_first(
        raw_spec,
        ("numeric_preprocessing", "follower_count_log_clip_upper"),
        ("preprocessing", "numeric_preprocessing", "follower_count_log_clip_upper"),
    )
    follower_clip_upper = _safe_float(follower_clip_upper, default=None)
    if "follower_count_log" not in clip_values and follower_clip_upper is not None:
        clip_values["follower_count_log"] = {"lower": None, "upper": float(follower_clip_upper)}

    theme_encoding = {
        "encoding_type": "smoothed_theme_target_mean_log1p",
        "target_space": "log1p_digg_count",
        "unseen_theme_fallback": "bayesian_global_mean",
        "global_mean_log": float(bayesian_global_mean),
        "smoothing_weight": float(smoothing_weight),
        "theme_stats": theme_stats,
        "theme_statistics": theme_stats,
    }

    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "derivation_version": str(
            raw_spec.get("derivation_version") or "normalized_preprocessing_spec_v2"
        ),
        "version_id": source_version,
        "created_at": raw_spec.get("created_at"),
        "producer": raw_spec.get("producer"),
        "expected_consumer_mode": raw_spec.get(
            "expected_consumer_mode", "version_owned_preprocessing_metadata"
        ),
        "feature_names_in": feature_order,
        "known_theme_cols": known_theme_cols,
        "bayesian_global_mean": float(bayesian_global_mean),
        "theme_encoding": theme_encoding,
        "numeric_imputations": numeric_imputations,
        "clip_values": clip_values,
        "preprocessing": {
            "feature_names_in": feature_order,
            "feature_order": feature_order,
            "known_theme_cols": known_theme_cols,
            "theme_encoding": theme_encoding,
            "numeric_imputations": numeric_imputations,
            "numeric_defaults": numeric_imputations,
            "clip_values": clip_values,
            "numeric_preprocessing": {
                "numeric_imputation_values": numeric_imputations,
                "follower_count_log_clip_upper": (
                    clip_values["follower_count_log"]["upper"]
                    if "follower_count_log" in clip_values
                    else None
                ),
            },
        },
    }


def preprocessing_context_from_spec(prep_spec):
    """Reconstruct a transformable preprocessing context from a canonical spec."""
    normalized = normalize_preprocessing_spec(prep_spec)
    if normalized is None:
        raise ValueError("Invalid preprocessing spec")

    theme_stats = normalized["theme_encoding"]["theme_stats"]
    fitted_theme_encoding = {
        "global_mean": float(normalized["bayesian_global_mean"]),
        "smoothing_weight": float(normalized["theme_encoding"]["smoothing_weight"]),
        "theme_count_by_theme": {},
        "theme_local_mean_by_theme": {},
        "theme_encoded_by_theme": {},
    }
    for theme_name, stats in theme_stats.items():
        fitted_theme_encoding["theme_count_by_theme"][theme_name] = int(stats.get("count", 0))
        if stats.get("local_mean_log") is not None:
            fitted_theme_encoding["theme_local_mean_by_theme"][theme_name] = float(
                stats["local_mean_log"]
            )
        fitted_theme_encoding["theme_encoded_by_theme"][theme_name] = float(
            stats.get("theme_encoded", normalized["bayesian_global_mean"])
        )

    follower_clip_upper = None
    if "follower_count_log" in normalized["clip_values"]:
        follower_clip_upper = normalized["clip_values"]["follower_count_log"].get("upper")
    if follower_clip_upper is None:
        follower_clip_upper = float("inf")

    return {
        "fitted_theme_encoding": fitted_theme_encoding,
        "known_theme_cols": list(normalized["known_theme_cols"]),
        "numeric_imputation_values": dict(normalized["numeric_imputations"]),
        "follower_clip_upper": float(follower_clip_upper),
        "feature_cols": list(normalized["feature_names_in"]),
        "prep_spec": normalized,
    }


def transform_with_preprocessing_context(df, preprocessing_context):
    """Apply a fitted preprocessing context to raw/eval data."""
    X = build_feature_matrix(
        df,
        preprocessing_context["fitted_theme_encoding"],
        preprocessing_context["known_theme_cols"],
    )
    X = apply_numeric_preprocessing(
        X,
        preprocessing_context["follower_clip_upper"],
        preprocessing_context["numeric_imputation_values"],
    )
    X = add_derived_features(X)
    return X[preprocessing_context["feature_cols"]]


def transform_with_preprocessing_spec(df, prep_spec):
    """Apply a serialized preprocessing spec directly to raw/eval data."""
    preprocessing_context = preprocessing_context_from_spec(prep_spec)
    return transform_with_preprocessing_context(df, preprocessing_context)


def _build_legacy_compatible_feature_frame(df, expected_features, scaler=None, manifest_payload=None):
    """Legacy-only fallback that uses raw data and version metadata, never UI baseline."""
    working_df = df.copy()
    if "theme_label" not in working_df.columns:
        working_df["theme_label"] = "Unknown"
    working_df["theme_label"] = working_df["theme_label"].fillna("Unknown")

    if "follower_count_log" not in working_df.columns and "follower_count" in working_df.columns:
        working_df["follower_count_log"] = np.log1p(working_df["follower_count"].fillna(0.0))
    if "publish_hour" not in working_df.columns and "create_time" in working_df.columns:
        working_df["publish_hour"] = pd.to_datetime(working_df["create_time"], errors="coerce").dt.hour
    if "duration_sec" not in working_df.columns and "duration" in working_df.columns:
        working_df["duration_sec"] = pd.to_numeric(working_df["duration"], errors="coerce").fillna(0.0)

    global_mean_log = _safe_float(
        (manifest_payload or {}).get("bayesian_global_mean"),
        default=DEFAULT_THEME_GLOBAL_MEAN_LOG,
    )
    X = pd.DataFrame(0.0, index=working_df.index, columns=list(expected_features))

    if "follower_count_log" in X.columns:
        X.loc[:, "follower_count_log"] = working_df.get("follower_count_log", 0.0).fillna(0.0).astype(float)
    if "publish_hour" in X.columns:
        X.loc[:, "publish_hour"] = working_df.get("publish_hour", 0.0).fillna(0.0).astype(float)
    if "duration_sec" in X.columns:
        X.loc[:, "duration_sec"] = working_df.get("duration_sec", 0.0).fillna(0.0).astype(float)
    if "avg_sentiment" in X.columns:
        X.loc[:, "avg_sentiment"] = working_df.get("avg_sentiment", 0.5).fillna(0.5).astype(float)
    if "visual_brightness" in X.columns:
        X.loc[:, "visual_brightness"] = working_df.get("visual_brightness", 128.0).fillna(128.0).astype(float)
    if "visual_saturation" in X.columns:
        X.loc[:, "visual_saturation"] = working_df.get("visual_saturation", 100.0).fillna(100.0).astype(float)
    if "cut_frequency" in X.columns:
        X.loc[:, "cut_frequency"] = working_df.get("cut_frequency", 0.5).fillna(0.5).astype(float)
    if "audio_bpm" in X.columns:
        X.loc[:, "audio_bpm"] = working_df.get("audio_bpm", 110.0).fillna(110.0).astype(float)
    if "theme_encoded" in X.columns:
        X.loc[:, "theme_encoded"] = float(global_mean_log)

    if "visual_impact" in X.columns:
        X.loc[:, "visual_impact"] = X["visual_brightness"] * X["visual_saturation"] / 1000.0
    if "sensory_pace" in X.columns:
        X.loc[:, "sensory_pace"] = X["audio_bpm"] * X["cut_frequency"]
    if "sentiment_intensity" in X.columns:
        X.loc[:, "sentiment_intensity"] = abs(X["avg_sentiment"] - 0.5) * 2
    if "audio_visual_energy" in X.columns:
        X.loc[:, "audio_visual_energy"] = X["visual_brightness"] * X["audio_bpm"] / 1000.0
    if "content_density" in X.columns:
        X.loc[:, "content_density"] = X["cut_frequency"] / (X["duration_sec"] + 1)

    for col in expected_features:
        if not str(col).startswith("theme_"):
            continue
        theme_name = str(col).replace("theme_", "")
        X.loc[:, col] = (working_df["theme_label"].astype(str) == theme_name).astype(float)

    if "theme_Unknown" in X.columns:
        mapped_rows = X[[c for c in X.columns if str(c).startswith("theme_")]].sum(axis=1) > 0
        X.loc[~mapped_rows, "theme_Unknown"] = 1.0

    if scaler is not None and hasattr(scaler, "mean_"):
        for idx, col in enumerate(expected_features):
            if idx >= len(scaler.mean_):
                continue
            if not str(col).startswith("theme_") and col in X.columns:
                X.loc[:, col] = X[col].fillna(float(scaler.mean_[idx]))

    return X[list(expected_features)]


def build_versioned_evaluation_features(
    eval_df,
    expected_features,
    scaler=None,
    prep_spec=None,
    manifest_payload=None,
):
    """
    Build evaluation features from raw data using version-owned preprocessing truth when available.
    Falls back to an explicit legacy-compatible reconstruction path when the version has no prep spec.
    """
    if prep_spec:
        transformed = transform_with_preprocessing_spec(eval_df, prep_spec)
        transformed = transformed.reindex(columns=list(expected_features))
        if scaler is not None and hasattr(scaler, "mean_"):
            for idx, col in enumerate(expected_features):
                if idx >= len(scaler.mean_):
                    continue
                if not str(col).startswith("theme_") and col in transformed.columns:
                    transformed.loc[:, col] = transformed[col].fillna(float(scaler.mean_[idx]))
        transformed = transformed.fillna(0.0)
        return transformed[list(expected_features)]

    return _build_legacy_compatible_feature_frame(
        df=eval_df,
        expected_features=expected_features,
        scaler=scaler,
        manifest_payload=manifest_payload,
    )


def resolve_prep_spec_path(artifacts_dir, version_id, manifest_payload=None):
    """Resolve the prep-spec path using manifest linkage first, then the legacy naming convention."""
    artifacts_dir = Path(artifacts_dir)
    candidates = []

    if isinstance(manifest_payload, dict) and manifest_payload.get("current_version") == version_id:
        bundle_artifacts = manifest_payload.get("bundle_artifacts", {})
        for candidate_name in (
            bundle_artifacts.get("prep_spec_file"),
            manifest_payload.get("prep_spec_file"),
        ):
            if not candidate_name:
                continue
            candidate_path = Path(candidate_name)
            if not candidate_path.is_absolute():
                candidate_path = artifacts_dir / candidate_path
            candidates.append(candidate_path)

    candidates.append(artifacts_dir / f"prep_{version_id}.json")

    seen = set()
    deduped = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    for path in deduped:
        if path.exists():
            return path

    return deduped[0] if deduped else artifacts_dir / f"prep_{version_id}.json"
