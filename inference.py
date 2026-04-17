from __future__ import annotations

import json
import os
import re
import sys
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT

DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "comparisons" / "best_original_intensive_v1_grid_lr.joblib"
DEFAULT_CLASS_MAPPING_PATH = WORKSPACE_ROOT / "comparisons" / "class_mapping.csv"
DEFAULT_PATTERN_PATH = WORKSPACE_ROOT / "disease_patterns.json"
DEFAULT_CUTOFF_XLSX = WORKSPACE_ROOT / "Edit20260116_KCMH_cut-off-Summary_N28778_48to72_Normals.xlsx"
DEFAULT_TRAIN_CSV = WORKSPACE_ROOT / "new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv"


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan


def _normalize_marker_text(value: object) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _build_marker_column_map(
    required_markers: List[str],
    available_columns: List[str],
    threshold: float = 1.0,
) -> Dict[str, str]:
    """Map marker names to real dataframe columns using strict similarity matching."""
    mapping: Dict[str, str] = {}
    if not required_markers or not available_columns:
        return mapping

    norm_cache = {col: _normalize_marker_text(col) for col in available_columns}
    for marker in required_markers:
        marker_str = str(marker)
        if marker_str in mapping:
            continue
        if marker_str in norm_cache:
            mapping[marker_str] = marker_str
            continue

        marker_norm = _normalize_marker_text(marker_str)
        best_col = None
        best_score = -1.0
        for col in available_columns:
            score = SequenceMatcher(None, marker_norm, norm_cache[col]).ratio()
            if score > best_score:
                best_score = score
                best_col = col

        if best_col is not None and best_score >= threshold:
            mapping[marker_str] = best_col

    return mapping


def _collect_required_marker_keys(
    matched_markers: List[str],
    parsed_marker_map: Dict[str, dict[str, object]],
) -> List[str]:
    keys: List[str] = []

    def _append_key(value: object) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for item in value:
                _append_key(item)
            return
        key_str = str(value).strip()
        if key_str:
            keys.append(key_str)

    for marker in matched_markers:
        _append_key(marker)
        parsed = parsed_marker_map.get(marker) or {}
        if parsed.get("type") == "single":
            _append_key(parsed.get("xml_markers"))
        if parsed.get("type") == "ratio":
            _append_key(parsed.get("numerator_xml"))
            _append_key(parsed.get("denominator_xml"))

    return list(dict.fromkeys(keys))


def _non_engineered_cols(columns: List[str]) -> List[str]:
    return [c for c in columns if not str(c).endswith(("_MoM", "_below_cutoff", "_above_cutoff"))]


class LogTransformer:
    """Compatibility shim for notebook-trained pipeline unpickling."""

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.columns_ = list(X.columns)
            base_cols = _non_engineered_cols(self.columns_)
            self.log_cols_ = [
                c for c in base_cols if np.all(pd.to_numeric(X[c], errors="coerce").fillna(0) >= 0)
            ]
        else:
            X_arr = np.array(X, dtype=float)
            self.non_negative_mask_ = np.all(X_arr >= 0, axis=0)
            self.log_cols_ = None
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X_out = X.copy()
            log_cols = getattr(self, "log_cols_", None) or []
            if log_cols:
                X_out = X_out.astype({c: float for c in log_cols}, copy=False)
                X_out.loc[:, log_cols] = np.log1p(X_out.loc[:, log_cols])
            return X_out

        X_arr = np.array(X, dtype=float)
        X_arr = X_arr.clip(min=-0.999999)
        X_out = X_arr.copy()
        X_out[:, self.non_negative_mask_] = np.log1p(X_out[:, self.non_negative_mask_])
        return X_out


class SelectiveRobustScaler:
    """Compatibility shim for notebook-trained pipeline unpickling."""

    def fit(self, X, y=None):
        self.scaler_ = RobustScaler()
        if hasattr(X, "columns"):
            self.scale_cols_ = _non_engineered_cols(list(X.columns))
            self.scaler_.fit(X[self.scale_cols_].astype(float))
        else:
            self.scale_cols_ = None
            self.scaler_.fit(X)
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X_out = X.copy()
            if self.scale_cols_:
                X_out = X_out.astype({c: float for c in self.scale_cols_}, copy=False)
                X_out.loc[:, self.scale_cols_] = self.scaler_.transform(X_out.loc[:, self.scale_cols_])
            return X_out
        return self.scaler_.transform(X)


def register_pickle_compat_classes() -> None:
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        if not hasattr(main_mod, "LogTransformer"):
            setattr(main_mod, "LogTransformer", LogTransformer)
        if not hasattr(main_mod, "SelectiveRobustScaler"):
            setattr(main_mod, "SelectiveRobustScaler", SelectiveRobustScaler)


def patch_legacy_model_attrs(model) -> None:
    """Backfill attributes sometimes missing in older pickled sklearn estimators."""
    if hasattr(model, "named_steps"):
        lr_step = model.named_steps.get("model")
        if lr_step is not None and lr_step.__class__.__name__ == "LogisticRegression":
            if not hasattr(lr_step, "multi_class"):
                lr_step.multi_class = "auto"
            if not hasattr(lr_step, "n_features_in_") and hasattr(model.named_steps.get("log"), "columns_"):
                lr_step.n_features_in_ = len(model.named_steps["log"].columns_)


def sanitize_feature_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "feature"


def make_unique(names: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        out.append(n if counts[n] == 1 else f"{n}_{counts[n]}")
    return out


def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = make_unique([sanitize_feature_name(c) for c in out.columns])
    return out


def detect_id_column(df: pd.DataFrame, user_id_col: Optional[str]) -> str:
    if user_id_col is not None:
        if user_id_col not in df.columns:
            raise ValueError(f"Requested id-column '{user_id_col}' not found")
        return user_id_col

    candidates = ["Barcode", "LabNumber", "ID", "id", "sample_id", "SampleID", "HN"]
    for col in candidates:
        if col in df.columns:
            return col

    df["row_id"] = np.arange(1, len(df) + 1)
    return "row_id"


def normalize_sample_id(value) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        return text[2:-1]
    return text


def load_class_mapping(mapping_csv: Path) -> Dict[int, str]:
    if not mapping_csv.exists():
        return {}
    mapping_df = pd.read_csv(mapping_csv)
    required = {"class_id", "class_name"}
    if not required.issubset(mapping_df.columns):
        return {}
    return dict(zip(mapping_df["class_id"], mapping_df["class_name"]))


def resolve_disease_names(model, class_mapping_csv: Path) -> List[str]:
    class_ids = list(model.classes_)
    mapping = load_class_mapping(class_mapping_csv)
    if mapping:
        return [str(mapping.get(cid, f"class_{cid}")) for cid in class_ids]
    if all(isinstance(c, str) for c in class_ids):
        return [str(c) for c in class_ids]
    return [f"class_{cid}" for cid in class_ids]


def get_expected_features(model) -> Optional[List[str]]:
    if hasattr(model, "named_steps"):
        log_step = model.named_steps.get("log")
        if hasattr(log_step, "columns_"):
            return list(log_step.columns_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def align_features(df: pd.DataFrame, expected_features: Optional[List[str]]) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c) for c in data.columns]

    if expected_features is None:
        return data.apply(pd.to_numeric, errors="coerce").fillna(0)

    missing = [c for c in expected_features if c not in data.columns]
    if missing:
        fill_block = pd.DataFrame(0, index=data.index, columns=missing)
        data = pd.concat([data, fill_block], axis=1)

    data = data[expected_features]
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0)
    return data


@dataclass
class FeatureAssets:
    median_map: Dict[str, float]
    cutoff_map: Dict[str, Dict[str, float]]
    pattern_df: Optional[pd.DataFrame]


def load_feature_engineering_assets(
    pattern_path: Path = DEFAULT_PATTERN_PATH,
    cutoff_excel: Path = DEFAULT_CUTOFF_XLSX,
) -> FeatureAssets:
    median_map: Dict[str, float] = {}
    cutoff_map: Dict[str, Dict[str, float]] = {}
    pattern_df: Optional[pd.DataFrame] = None

    if cutoff_excel.exists():
        median_df = pd.read_excel(cutoff_excel, sheet_name="Median")
        median_df.columns = ["marker", "median"]
        cutoff_df = pd.read_excel(cutoff_excel, sheet_name="Cut-off")
        cutoff_df.columns = ["marker", "lower_cutoff", "upper_cutoff"]
        median_df["marker"] = median_df["marker"].astype(str)
        cutoff_df["marker"] = cutoff_df["marker"].astype(str)
        median_map_raw = median_df.set_index("marker")["median"].to_dict()
        cutoff_map_raw = cutoff_df.set_index("marker")[["lower_cutoff", "upper_cutoff"]].to_dict(orient="index")
        median_map = {str(k): float(v) for k, v in median_map_raw.items() if pd.notna(v)}
        cutoff_map = {
            str(k): {
                "lower_cutoff": _safe_float(v.get("lower_cutoff")),
                "upper_cutoff": _safe_float(v.get("upper_cutoff")),
            }
            for k, v in cutoff_map_raw.items()
        }

    if pattern_path.exists():
        with open(pattern_path, encoding="utf-8") as f:
            disease_patterns = json.load(f)
        pattern_df = pd.DataFrame(disease_patterns)

        disease_name_map = {
            "Isovaleric acidemia (IVA)": "Isovaleric acidemia",
            "Propionic acidemia (PA)": "Propionic acidemia",
            "3-Hydroxy-3-Methylglutaryl-CoA (HMG-CoA) lyase deficiency": "3-Hydroxy-3-Methylglutaryl-CoA lyase deficiency",
            "Short-chain acyl-CoA dehydrogenase (SCAD) deficiency": "Short-chain acyl-CoA dehydrogenase deficiency",
            "Primary systemic carnitine deficiency (Carnitine uptake defect, CUD)": "Carnitine uptake defect / carnitine transport defect (plasma membrane carnitine transporter)",
            "Citrullinemia type 2 or Citrin deficiency": "Citrullinemia type II",
            "Tyrosinemia type 1 (TYR1)": "Tyrosinemia type I (fumarylacetoacetate hydrolase)",
        }

        pattern_df["DiseaseName_mapped"] = pattern_df["DiseaseName"].map(
            lambda x: disease_name_map.get(str(x), str(x)) if pd.notna(x) else x
        )

    return FeatureAssets(median_map=median_map, cutoff_map=cutoff_map, pattern_df=pattern_df)


def build_marker_weights(train_csv: Optional[Path], pattern_df: Optional[pd.DataFrame], gain: float = 0.50) -> Dict[str, float]:
    if pattern_df is None or train_csv is None or not train_csv.exists():
        return {}

    train_df = pd.read_csv(train_csv)
    if "EnzymeDefect" not in train_df.columns:
        return {}

    _, y_train = train_test_split(
        train_df["EnzymeDefect"].astype(str),
        test_size=0.2,
        stratify=train_df["EnzymeDefect"].astype(str),
        random_state=42,
    )

    class_counts = y_train.value_counts().to_dict()
    n_train = len(y_train)

    disease_to_markers: Dict[str, List[str]] = {}
    grouped = pattern_df.dropna(subset=["DiseaseName_mapped"]).groupby("DiseaseName_mapped")
    for dname, g in grouped:
        markers = pd.unique(g["xml_name"].dropna().astype(str)).tolist()
        if markers:
            disease_to_markers[str(dname)] = sorted(set(markers))

    marker_weights: Dict[str, float] = {}
    for dname, markers in disease_to_markers.items():
        cnt = int(class_counts.get(dname, 1))
        rarity_boost = np.log1p(n_train / max(cnt, 1))
        weight = 1.0 + gain * rarity_boost
        for m in markers:
            marker_weights[m] = max(marker_weights.get(m, 1.0), weight)

    return marker_weights


def _coerce_row_numeric(
    row: pd.Series,
    key: object,
    marker_column_map: Optional[Dict[str, str]] = None,
) -> float:
    mapped_key = key
    if marker_column_map is not None and key is not None:
        mapped_key = marker_column_map.get(str(key), key)
    return pd.to_numeric(pd.Series([row.get(mapped_key)]), errors="coerce").iloc[0]


def _resolve_parsed_operand_value(
    row: pd.Series,
    operand: object,
    operand_type: Optional[str],
    marker_column_map: Optional[Dict[str, str]] = None,
) -> float:
    if operand_type == "single":
        return _coerce_row_numeric(row, operand, marker_column_map)

    if operand_type == "sum":
        keys = operand if isinstance(operand, list) else [operand]
        values: list[float] = []
        for key in keys:
            value = _coerce_row_numeric(row, key, marker_column_map)
            if pd.isna(value):
                return np.nan
            values.append(float(value))
        return float(sum(values)) if values else np.nan

    return np.nan


def _compute_marker_value(
    row: pd.Series,
    marker: str,
    parsed_marker_map: Dict[str, dict[str, object]],
    marker_column_map: Optional[Dict[str, str]] = None,
) -> float:
    direct_value = _coerce_row_numeric(row, marker, marker_column_map)
    if pd.notna(direct_value):
        return direct_value

    parsed = parsed_marker_map.get(marker) or {}
    parsed_type = parsed.get("type")
    if parsed_type == "single":
        xml_markers = parsed.get("xml_markers")
        keys = xml_markers if isinstance(xml_markers, list) else [xml_markers]
        keys = [key for key in keys if key]
        if len(keys) != 1:
            return np.nan
        return _coerce_row_numeric(row, keys[0], marker_column_map)

    if parsed_type == "ratio":
        numerator = _resolve_parsed_operand_value(
            row,
            parsed.get("numerator_xml"),
            parsed.get("numerator_type"),
            marker_column_map,
        )
        denominator = _resolve_parsed_operand_value(
            row,
            parsed.get("denominator_xml"),
            parsed.get("denominator_type"),
            marker_column_map,
        )
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return np.nan
        return float(numerator) / float(denominator)

    return np.nan


def build_marker_detail_table(
    original_df: pd.DataFrame,
    id_col: str,
    median_map: Dict[str, float],
    cutoff_map: Dict[str, Dict[str, float]],
    pattern_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    parsed_marker_map: Dict[str, dict[str, object]] = {}
    if pattern_df is not None and not pattern_df.empty and {"xml_name", "parsed"}.issubset(pattern_df.columns):
        pattern_subset = pattern_df.dropna(subset=["xml_name"])
        for _, pattern_row in pattern_subset.iterrows():
            marker_name = str(pattern_row["xml_name"])
            if marker_name not in parsed_marker_map:
                parsed_value = pattern_row.get("parsed")
                parsed_marker_map[marker_name] = parsed_value if isinstance(parsed_value, dict) else {}

    matched_markers = list(
        dict.fromkeys([*median_map.keys(), *cutoff_map.keys(), *parsed_marker_map.keys()])
    )
    required_keys = _collect_required_marker_keys(matched_markers, parsed_marker_map)
    available_columns = [str(col) for col in original_df.columns]
    marker_column_map = _build_marker_column_map(required_keys, available_columns, threshold=1.0)

    for row_pos, (_, row) in enumerate(original_df.iterrows()):
        sample_id = normalize_sample_id(row[id_col])
        for marker in matched_markers:
            raw = _compute_marker_value(row, marker, parsed_marker_map, marker_column_map)
            if pd.isna(raw):
                continue

            med = median_map.get(marker)
            mom = np.nan
            if pd.notna(raw) and pd.notna(med) and med not in (0, 0.0):
                mom = raw / float(med)

            co = cutoff_map.get(marker, {})
            lo = co.get("lower_cutoff")
            hi = co.get("upper_cutoff")
            below = int(pd.notna(raw) and pd.notna(lo) and raw < lo)
            above = int(pd.notna(raw) and pd.notna(hi) and raw > hi)

            rows.append(
                {
                    "sample_id": sample_id,
                    "row_index": int(row_pos),
                    "marker": marker,
                    "value": None if pd.isna(raw) else float(raw),
                    "MoM": None if pd.isna(mom) else float(mom),
                    "lower_cutoff": None if pd.isna(lo) else float(lo),
                    "upper_cutoff": None if pd.isna(hi) else float(hi),
                    "below_cutoff": below,
                    "above_cutoff": above,
                }
            )

    return pd.DataFrame(rows)


def apply_training_feature_engineering(
    input_df: pd.DataFrame,
    id_col: str,
    assets: FeatureAssets,
    marker_weights: Dict[str, float],
) -> pd.DataFrame:
    work_df = input_df.copy()

    if "Abnormals" in work_df.columns:
        work_df["Abnormals"] = work_df["Abnormals"].notna().astype(int)

    if "C5/C0" not in work_df.columns and {"C5", "C0"}.issubset(work_df.columns):
        work_df["C5/C0"] = pd.to_numeric(work_df["C5"], errors="coerce") / pd.to_numeric(
            work_df["C0"], errors="coerce"
        ).replace({0: np.nan})

    drop_cols = [
        c
        for c in [
            "EnzymeDefect",
            "EnzymeDefect_enc",
            id_col,
            "LabNumber",
            "ReportAbnormals",
            "DoctorAppvDate",
        ]
        if c in work_df.columns
    ]

    X = work_df.drop(columns=drop_cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    for marker, med in assets.median_map.items():
        if marker not in X.columns:
            continue

        if pd.notna(med) and med != 0:
            X[f"{marker}_MoM"] = X[marker] / med
        else:
            X[f"{marker}_MoM"] = 1.0

        co = assets.cutoff_map.get(marker, {})
        lo = co.get("lower_cutoff")
        hi = co.get("upper_cutoff")
        if pd.notna(lo):
            X[f"{marker}_below_cutoff"] = (X[marker] < lo).astype(int)
        if pd.notna(hi):
            X[f"{marker}_above_cutoff"] = (X[marker] > hi).astype(int)

    for marker, weight in marker_weights.items():
        if marker in X.columns:
            X[marker] = X[marker] * weight
        for suffix in ("_MoM", "_below_cutoff", "_above_cutoff"):
            name = f"{marker}{suffix}"
            if name in X.columns:
                X[name] = X[name] * weight

    X = X.replace([np.inf, -np.inf], np.nan)
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values).fillna(0)
    X = sanitize_dataframe_columns(X)
    return X


def predict_from_dataframe(
    input_df: pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
    class_mapping_path: Path = DEFAULT_CLASS_MAPPING_PATH,
    train_csv_path: Optional[Path] = DEFAULT_TRAIN_CSV,
    id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    register_pickle_compat_classes()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    patch_legacy_model_attrs(model)
    source_df = input_df.copy()
    id_col = detect_id_column(source_df, id_column)

    assets = load_feature_engineering_assets()
    marker_weights = build_marker_weights(train_csv_path, assets.pattern_df)
    feature_df = apply_training_feature_engineering(source_df, id_col, assets, marker_weights)

    expected_features = get_expected_features(model)
    X = align_features(feature_df, expected_features)

    probs = model.predict_proba(X)
    top3_idx = np.argsort(-probs, axis=1)[:, :3]

    disease_names = resolve_disease_names(model, class_mapping_path)
    class_ids = list(model.classes_)

    rows = []
    for i in range(len(source_df)):
        sample_id = normalize_sample_id(source_df.iloc[i][id_col])
        i1, i2, i3 = top3_idx[i]
        rows.append(
            {
                "sample_id": sample_id,
                "top_1_class_id": class_ids[i1],
                "top_1_disease": disease_names[i1],
                "top_1_probability": float(probs[i, i1]),
                "top_2_class_id": class_ids[i2],
                "top_2_disease": disease_names[i2],
                "top_2_probability": float(probs[i, i2]),
                "top_3_class_id": class_ids[i3],
                "top_3_disease": disease_names[i3],
                "top_3_probability": float(probs[i, i3]),
            }
        )

    predictions_df = pd.DataFrame(rows)
    marker_detail_df = build_marker_detail_table(source_df, id_col, assets.median_map, assets.cutoff_map, assets.pattern_df)

    return predictions_df, marker_detail_df, id_col
