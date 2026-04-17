from __future__ import annotations

import os
import re
from datetime import datetime
from html import escape
from pathlib import Path
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from dotenv import load_dotenv

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from inference import (
    DEFAULT_CLASS_MAPPING_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_CSV,
    load_feature_engineering_assets,
    predict_from_dataframe,
)


load_dotenv()

ROOT = Path(__file__).resolve().parent


def is_normal_disease_name(name: str) -> bool:
    text = str(name).strip().lower()
    return text == "normal/other unspecified disease" or text.startswith("normal")


def is_control_or_internal_sample(sample_id: str) -> bool:
    text = str(sample_id).strip().upper()
    if not text:
        return False
    return text.startswith("LC") or text.startswith("HC") or text == "IS"


def _prepare_marker_candidates(marker_detail_df: pd.DataFrame) -> pd.DataFrame:
    if marker_detail_df.empty:
        return marker_detail_df.copy()

    work = marker_detail_df.copy()
    for col in ["value", "MoM", "lower_cutoff", "upper_cutoff", "below_cutoff", "above_cutoff"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["cutoff_abnormal"] = (
        (work["below_cutoff"].fillna(0) > 0) | (work["above_cutoff"].fillna(0) > 0)
    ).astype(int)
    work["mom_sort"] = work["MoM"].fillna(-np.inf)

    mom_for_tie = work["MoM"].where(work["MoM"] > 0)
    mom_distance = pd.Series(np.log2(mom_for_tie), index=work.index)
    work["mom_distance"] = mom_distance.abs().replace([np.inf, -np.inf], np.nan).fillna(0)
    return work


def _lookup_disease_markers(
    disease_name: str,
    disease_to_markers: dict[str, list[str]] | None,
) -> list[str]:
    if not disease_to_markers:
        return []
    markers = disease_to_markers.get(str(disease_name), [])
    if markers:
        return markers
    return disease_to_markers.get(_canonicalize_disease_name(disease_name), [])


def _select_marker_by_triage(
    sample_marker_df: pd.DataFrame,
    preferred_markers: list[str] | None = None,
) -> pd.Series | None:
    if sample_marker_df.empty:
        return None

    prepared = _prepare_marker_candidates(sample_marker_df)
    preferred_markers = preferred_markers or []

    candidate_sets: list[pd.DataFrame] = []
    if preferred_markers:
        preferred_df = prepared[prepared["marker"].astype(str).isin(preferred_markers)].copy()
        abnormal_preferred_df = preferred_df[preferred_df["cutoff_abnormal"] == 1].copy()
        if not abnormal_preferred_df.empty:
            candidate_sets.append(abnormal_preferred_df)
        if not preferred_df.empty:
            candidate_sets.append(preferred_df)
    abnormal_df = prepared[prepared["cutoff_abnormal"] == 1].copy()
    if not abnormal_df.empty:
        candidate_sets.append(abnormal_df)
    candidate_sets.append(prepared)

    for candidates in candidate_sets:
        ranked = candidates.sort_values(
            by=["cutoff_abnormal", "mom_distance", "mom_sort"],
            ascending=[False, False, False],
            kind="stable",
        )
        if not ranked.empty:
            return ranked.iloc[0]
    return None


def pick_top_marker_per_sample(
    marker_detail_df: pd.DataFrame,
    prioritized_df: pd.DataFrame,
    disease_to_markers: dict[str, list[str]] | None,
) -> pd.DataFrame:
    top_rows: list[dict[str, object]] = []

    if marker_detail_df.empty or prioritized_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "marker",
                "value",
                "MoM",
                "lower_cutoff",
                "upper_cutoff",
                "cutoff_status",
            ]
        )

    for _, row in prioritized_df.iterrows():
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue

        sample_marker_df = marker_detail_df[
            marker_detail_df["sample_id"].astype(str) == sample_id
        ].copy()
        preferred_markers = _lookup_disease_markers(
            str(row.get("top_1_disease", "")),
            disease_to_markers,
        )
        best = _select_marker_by_triage(sample_marker_df, preferred_markers)

        if best is None:
            top_rows.append(
                {
                    "sample_id": sample_id,
                    "marker": "-",
                    "value": np.nan,
                    "MoM": np.nan,
                    "lower_cutoff": np.nan,
                    "upper_cutoff": np.nan,
                    "cutoff_status": "-",
                }
            )
            continue

        below_value = pd.to_numeric(pd.Series([best.get("below_cutoff")]), errors="coerce").iloc[0]
        above_value = pd.to_numeric(pd.Series([best.get("above_cutoff")]), errors="coerce").iloc[0]
        below = bool(pd.notna(below_value) and float(below_value) > 0)
        above = bool(pd.notna(above_value) and float(above_value) > 0)
        if below and above:
            cutoff_status = "below lower and above upper"
        elif below:
            cutoff_status = "below lower"
        elif above:
            cutoff_status = "above upper"
        else:
            cutoff_status = "within cut-off"

        top_rows.append(
            {
                "sample_id": sample_id,
                "marker": best.get("marker", "-"),
                "value": pd.to_numeric(pd.Series([best.get("value")]), errors="coerce").iloc[0],
                "MoM": pd.to_numeric(pd.Series([best.get("MoM")]), errors="coerce").iloc[0],
                "lower_cutoff": pd.to_numeric(pd.Series([best.get("lower_cutoff")]), errors="coerce").iloc[0],
                "upper_cutoff": pd.to_numeric(pd.Series([best.get("upper_cutoff")]), errors="coerce").iloc[0],
                "cutoff_status": cutoff_status,
            }
        )

    return pd.DataFrame(top_rows).reset_index(drop=True)


def build_suspected_case_html_report(suspected_df: pd.DataFrame) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_mom(value: object) -> str:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "-" if pd.isna(v) else f"{float(v):.2f}"

    if suspected_df.empty:
        html = f"""
        <!DOCTYPE html>
        <html lang='en'>
        <head>
            <meta charset='utf-8' />
            <title>Suspected Disease Cases Report</title>
            <style>
                body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }}
                .title {{ background: #1e8f4e; color: #fff; padding: 12px; font-size: 20px; font-weight: 700; }}
                .ok {{ margin-top: 12px; padding: 10px; border: 1px solid #a6dcb7; background: #eefaf1; color: #18633a; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <div class='title'>Disease Pattern Detection Report (Suspected Cases Only)</div>
            <div class='ok'>No suspected disease cases found.</div>
            <div style='margin-top:10px;color:#666;font-size:12px;'>Generated: {generated_at}</div>
        </body>
        </html>
        """
        return html

    rows = []
    for _, row in suspected_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('sample_id', '-')))}</td>"
            f"<td>{escape(str(row.get('top_1_disease', '-')))}</td>"
            f"<td>{float(row.get('top_1_probability', 0.0))*100:.2f}%</td>"
            f"<td>{escape(str(row.get('marker', '-')))}</td>"
            f"<td>{escape(str(row.get('value', '-')))}</td>"
            f"<td>{_fmt_mom(row.get('MoM', '-'))}</td>"
            f"<td>{escape(str(row.get('lower_cutoff', '-')))}</td>"
            f"<td>{escape(str(row.get('upper_cutoff', '-')))}</td>"
            f"<td>{escape(str(row.get('top_2_disease', '-')))}</td>"
            f"<td>{float(row.get('top_2_probability', 0.0))*100:.2f}%</td>"
            f"<td>{escape(str(row.get('top_3_disease', '-')))}</td>"
            f"<td>{float(row.get('top_3_probability', 0.0))*100:.2f}%</td>"
            "</tr>"
        )

    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='utf-8' />
        <title>Suspected Disease Cases Report</title>
        <style>
            body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }}
            .title {{ background: #1e8f4e; color: #fff; padding: 12px; font-size: 20px; font-weight: 700; margin-bottom: 12px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #1e8f4e; color: #fff; border: 1px solid #16673a; padding: 8px; font-size: 13px; }}
            td {{ border: 1px solid #d9e5dd; padding: 8px; font-size: 13px; color: #b00020; }}
            tr:nth-child(1) td {{ background: #fdf1f4; font-weight: 700; }}
        </style>
    </head>
    <body>
        <div class='title'>Disease Pattern Detection Report (Suspected Cases Only)</div>
        <table>
            <thead>
                <tr>
                    <th>Sample ID</th><th>Top 1 Disease</th><th>Top 1 %</th>
                    <th>Top Marker</th><th>Marker Value</th><th>MoM</th>
                    <th>Lower Cut-off</th><th>Upper Cut-off</th>
                    <th>Top 2 Disease</th><th>Top 2 %</th>
                    <th>Top 3 Disease</th><th>Top 3 %</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <div style='margin-top:10px;color:#666;font-size:12px;'>Generated: {generated_at}</div>
    </body>
    </html>
    """
    return html


def build_disease_to_markers_map() -> dict[str, list[str]]:
    assets = load_feature_engineering_assets()
    pattern_df = assets.pattern_df
    if pattern_df is None or pattern_df.empty:
        return {}

    mapping: dict[str, list[str]] = {}
    grouped = pattern_df.dropna(subset=["DiseaseName_mapped"]).groupby("DiseaseName_mapped")
    for disease, g in grouped:
        markers = pd.unique(g["xml_name"].dropna().astype(str)).tolist()
        if markers:
            unique_markers = sorted(set(markers))
            raw_names = pd.unique(g["DiseaseName"].dropna().astype(str)).tolist() if "DiseaseName" in g.columns else []
            aliases = {str(disease)}
            aliases.update(raw_names)

            expanded_aliases = set()
            for alias in aliases:
                expanded_aliases.add(alias)
                expanded_aliases.add(_canonicalize_disease_name(alias))
                alias_no_paren = re.sub(r"\s*\([^)]*\)", "", alias).strip()
                if alias_no_paren:
                    expanded_aliases.add(alias_no_paren)
                    expanded_aliases.add(_canonicalize_disease_name(alias_no_paren))
                for splitter in [" or ", "/", ","]:
                    if splitter in alias.lower():
                        parts = [part.strip() for part in re.split(r"\s+or\s+|/|,", alias) if part.strip()]
                        for part in parts:
                            expanded_aliases.add(part)
                            expanded_aliases.add(_canonicalize_disease_name(part))

            for alias in expanded_aliases:
                if alias:
                    mapping[str(alias)] = unique_markers
    return mapping


def load_trained_disease_names(class_mapping_path: Path) -> set[str]:
    if not class_mapping_path.exists():
        return set()

    mapping_df = pd.read_csv(class_mapping_path)
    if "class_name" not in mapping_df.columns:
        return set()
    return set(mapping_df["class_name"].dropna().astype(str))


def build_additional_pattern_support_report(
    marker_detail_df: pd.DataFrame,
    class_mapping_path: Path,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "disease_name",
        "pattern_support_percent",
        "abnormal_marker_count",
        "available_pattern_marker_count",
        "total_pattern_marker_count",
        "available_pattern_markers",
        "abnormal_markers",
        "representative_marker",
        "value",
        "MoM",
        "lower_cutoff",
        "upper_cutoff",
        "cutoff_status",
    ]
    if marker_detail_df.empty:
        return pd.DataFrame(columns=columns)

    assets = load_feature_engineering_assets()
    pattern_df = assets.pattern_df
    if pattern_df is None or pattern_df.empty or "DiseaseName" not in pattern_df.columns:
        return pd.DataFrame(columns=columns)

    trained_diseases = load_trained_disease_names(class_mapping_path)
    trained_canonical = {_canonicalize_disease_name(name) for name in trained_diseases}

    disease_groups: list[tuple[str, str, str, list[str]]] = []
    grouped = pattern_df.dropna(subset=["DiseaseName"]).groupby("DiseaseName")
    for raw_name, group in grouped:
        markers = sorted(set(group["xml_name"].dropna().astype(str)))
        if not markers:
            continue
        mapped_names = group["DiseaseName_mapped"].dropna().astype(str).tolist() if "DiseaseName_mapped" in group.columns else []
        mapped_name = mapped_names[0] if mapped_names else str(raw_name)
        raw_aliases = _expand_disease_aliases(str(raw_name))
        mapped_aliases = _expand_disease_aliases(mapped_name)
        if raw_aliases & trained_canonical:
            continue
        if mapped_aliases & trained_canonical:
            continue
        display_name = re.sub(r"\s*\(\d+\)\s*$", "", str(raw_name)).strip()
        disease_groups.append((str(raw_name), display_name, mapped_name, markers))

    if not disease_groups:
        return pd.DataFrame(columns=columns)

    prepared_markers = _prepare_marker_candidates(marker_detail_df)
    rows: list[dict[str, object]] = []

    for sample_id, sample_marker_df in prepared_markers.groupby("sample_id", sort=True):
        sample_marker_df = sample_marker_df.copy()
        for raw_name, display_name, _mapped_name, markers in disease_groups:
            disease_marker_df = sample_marker_df[sample_marker_df["marker"].astype(str).isin(markers)].copy()
            available_marker_names = sorted(set(disease_marker_df["marker"].astype(str)))
            available_marker_count = len(available_marker_names)
            if available_marker_count == 0:
                continue

            # CUD: only below-cutoff markers count as abnormal
            _raw_lower = str(raw_name).lower()
            is_cud = "carnitine uptake" in _raw_lower or ("cud" in _raw_lower and "carnitine" in _raw_lower)
            if is_cud:
                abnormal_df = disease_marker_df[pd.to_numeric(disease_marker_df.get("below_cutoff", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0].copy()
            else:
                abnormal_df = disease_marker_df[disease_marker_df["cutoff_abnormal"] == 1].copy()
            abnormal_marker_names = sorted(set(abnormal_df["marker"].astype(str)))
            abnormal_count = len(abnormal_marker_names)
            total_pattern_marker_count = len(markers)

            support_ratio = abnormal_count / available_marker_count
            if support_ratio <= 0.3:
                continue

            representative = pick_marker_for_disease(sample_marker_df, markers)
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "disease_name": display_name,
                    "pattern_support_percent": round(support_ratio * 100, 2),
                    "abnormal_marker_count": abnormal_count,
                    "available_pattern_marker_count": available_marker_count,
                    "total_pattern_marker_count": total_pattern_marker_count,
                    "available_pattern_markers": ", ".join(available_marker_names),
                    "abnormal_markers": ", ".join(abnormal_marker_names),
                    "representative_marker": representative["marker"],
                    "value": representative["value"],
                    "MoM": representative["mom"],
                    "lower_cutoff": representative["lower_cutoff"],
                    "upper_cutoff": representative["upper_cutoff"],
                    "cutoff_status": representative["cutoff_status"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        by=["sample_id", "pattern_support_percent", "abnormal_marker_count", "disease_name"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _canonicalize_disease_name(name: str) -> str:
    text = str(name).strip().lower()
    replacements = {
        "type iii": "type 3",
        "type ii": "type 2",
        "type iv": "type 4",
        "type i": "type 1",
        "citrin deficiency": "citrullinemia type 2",
        "primary systemic carnitine deficiency": "carnitine uptake defect",
        "cud": "carnitine uptake defect",
        "carnitine transport defect": "carnitine uptake defect",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_disease_aliases(name: str) -> set[str]:
    text = str(name).strip()
    if not text:
        return set()

    aliases = {text, re.sub(r"\s*\([^)]*\)", "", text).strip()}
    for group in re.findall(r"\(([^)]*)\)", text):
        group = group.strip()
        if group:
            aliases.add(group)

    expanded: set[str] = set()
    for alias in aliases:
        if not alias:
            continue
        expanded.add(_canonicalize_disease_name(alias))
        for part in re.split(r"\s+or\s+|/|,", alias):
            part = part.strip()
            if part:
                expanded.add(_canonicalize_disease_name(part))

    return {alias for alias in expanded if alias}


def _format_cutoff_value(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "-" if pd.isna(numeric) else f"{float(numeric):.4f}"


def pick_marker_for_disease(sample_marker_df: pd.DataFrame, disease_markers: list[str]) -> dict[str, str]:
    best = _select_marker_by_triage(sample_marker_df, disease_markers)
    if best is None:
        return {
            "marker": "-",
            "value": "-",
            "mom": "-",
            "lower_cutoff": "-",
            "upper_cutoff": "-",
            "cutoff_status": "-",
        }

    marker = str(best.get("marker", "-") or "-")
    val = best.get("value")
    mom = best.get("MoM")
    lo = best.get("lower_cutoff")
    hi = best.get("upper_cutoff")
    below_value = pd.to_numeric(pd.Series([best.get("below_cutoff")]), errors="coerce").iloc[0]
    above_value = pd.to_numeric(pd.Series([best.get("above_cutoff")]), errors="coerce").iloc[0]
    below = bool(pd.notna(below_value) and float(below_value) > 0)
    above = bool(pd.notna(above_value) and float(above_value) > 0)

    exceeded_parts: list[str] = []
    if below and pd.notna(lo):
        exceeded_parts.append(f"below lower ({float(lo):.4f})")
    if above and pd.notna(hi):
        exceeded_parts.append(f"above upper ({float(hi):.4f})")

    return {
        "marker": marker,
        "value": "-" if pd.isna(val) else f"{float(val):.4f}",
        "mom": "-" if pd.isna(mom) else f"{float(mom):.2f}",
        "lower_cutoff": _format_cutoff_value(lo),
        "upper_cutoff": _format_cutoff_value(hi),
        "cutoff_status": ", ".join(exceeded_parts) if exceeded_parts else "within cut-off",
    }


def _normalize_marker_key(marker: object) -> str:
    text = str(marker).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _build_flagged_3d_axis_markers(
    main_marker: str,
    suspected_disease: str,
    disease_to_markers: dict[str, list[str]],
    available_markers: list[str],
) -> list[str]:
    main_marker = str(main_marker).strip()
    if not main_marker:
        return []

    available_set = {str(m) for m in available_markers if str(m).strip()}
    if main_marker not in available_set:
        return []

    disease_markers = [m for m in _lookup_disease_markers(suspected_disease, disease_to_markers) if str(m) in available_set]
    norm_main = _normalize_marker_key(main_marker)

    related_ratio_markers: list[str] = []
    fallback_markers: list[str] = []
    for marker in disease_markers:
        marker = str(marker)
        if marker == main_marker:
            continue
        if "/" in marker and norm_main and norm_main in _normalize_marker_key(marker):
            related_ratio_markers.append(marker)
        else:
            fallback_markers.append(marker)

    selected: list[str] = []
    for marker in related_ratio_markers + fallback_markers:
        if marker not in selected:
            selected.append(marker)
        if len(selected) == 2:
            break

    axes = [main_marker] + selected
    return axes if len(axes) == 3 else []


def _draw_cutoff_line_3d(
    ax,
    axis_name: str,
    cutoff_value: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    color: str,
    marker_label: str,
    cutoff_kind: str,
) -> None:
    _ = marker_label
    y_mid = (y_range[0] + y_range[1]) / 2.0
    x_mid = (x_range[0] + x_range[1]) / 2.0
    line_style = dict(
        linestyle="--" if str(cutoff_kind).lower() == "lower" else ":",
        color=color,
        linewidth=1.35,
        alpha=0.86,
    )

    if axis_name == "x":
        # Draw one full-span line along Z on the x=cutoff plane.
        ax.plot(
            [cutoff_value, cutoff_value],
            [y_mid, y_mid],
            [z_range[0], z_range[1]],
            **line_style,
        )
        return
    if axis_name == "y":
        # Draw one full-span line along Z on the y=cutoff plane.
        ax.plot(
            [x_mid, x_mid],
            [cutoff_value, cutoff_value],
            [z_range[0], z_range[1]],
            **line_style,
        )
        return
    # Draw one full-span line along X on the z=cutoff plane.
    ax.plot(
        [x_range[0], x_range[1]],
        [y_mid, y_mid],
        [cutoff_value, cutoff_value],
        **line_style,
    )


def _build_plotly_cutoff_line(
    axis_name: str,
    cutoff_value: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[list[float], list[float], list[float]]:
    y_mid = (y_range[0] + y_range[1]) / 2.0
    x_mid = (x_range[0] + x_range[1]) / 2.0
    if axis_name == "x":
        return [cutoff_value, cutoff_value], [y_mid, y_mid], [z_range[0], z_range[1]]
    if axis_name == "y":
        return [x_mid, x_mid], [cutoff_value, cutoff_value], [z_range[0], z_range[1]]
    return [x_range[0], x_range[1]], [y_mid, y_mid], [cutoff_value, cutoff_value]


def _build_default_flagged_3d_png(
    prioritized_df: pd.DataFrame,
    patient_flagged_df: pd.DataFrame,
    suspected_top_marker_df: pd.DataFrame,
    marker_detail_df: pd.DataFrame,
    disease_to_markers: dict[str, list[str]],
) -> BytesIO | None:
    """Build a default static 3D graph PNG for the first flagged sample."""
    if patient_flagged_df.empty or marker_detail_df.empty:
        return None

    selected_flagged_id = str(patient_flagged_df.iloc[0]["sample_id"])
    selected_row_df = prioritized_df[prioritized_df["sample_id"].astype(str) == selected_flagged_id]
    if selected_row_df.empty:
        return None
    selected_disease = str(selected_row_df.iloc[0].get("top_1_disease", "-"))

    selected_top_marker_df = suspected_top_marker_df[
        suspected_top_marker_df["sample_id"].astype(str) == selected_flagged_id
    ]
    selected_main_marker = (
        str(selected_top_marker_df.iloc[0].get("marker", "")).strip()
        if not selected_top_marker_df.empty
        else ""
    )
    if not selected_main_marker:
        return None

    available_markers = sorted(set(marker_detail_df["marker"].dropna().astype(str)))
    axis_markers = _build_flagged_3d_axis_markers(
        main_marker=selected_main_marker,
        suspected_disease=selected_disease,
        disease_to_markers=disease_to_markers,
        available_markers=available_markers,
    )
    if len(axis_markers) != 3:
        return None

    work = _prepare_marker_candidates(marker_detail_df)
    keep_cols = ["sample_id", "marker", "value", "MoM", "lower_cutoff", "upper_cutoff"]
    work = work[[c for c in keep_cols if c in work.columns]].copy()
    work = work[work["marker"].astype(str).isin(axis_markers)].copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["MoM"] = pd.to_numeric(work["MoM"], errors="coerce")
    work["lower_cutoff"] = pd.to_numeric(work["lower_cutoff"], errors="coerce")
    work["upper_cutoff"] = pd.to_numeric(work["upper_cutoff"], errors="coerce")

    marker_ranked = work.sort_values(
        by=["sample_id", "marker", "MoM"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(subset=["sample_id", "marker"], keep="first")

    value_pivot = marker_ranked.pivot(index="sample_id", columns="marker", values="value")
    normal_ids = set(
        prioritized_df.loc[
            (~prioritized_df["is_flagged"]) & (~prioritized_df["is_control"]),
            "sample_id",
        ].astype(str)
    )
    normal_points = value_pivot[value_pivot.index.isin(normal_ids)].copy()
    normal_points = normal_points.dropna(subset=axis_markers, how="any")

    selected_point = value_pivot[value_pivot.index == selected_flagged_id].copy()
    selected_point = selected_point.dropna(subset=axis_markers, how="any")
    if selected_point.empty:
        return None

    x_marker, y_marker, z_marker = axis_markers
    x_val = float(selected_point.iloc[0][x_marker])
    y_val = float(selected_point.iloc[0][y_marker])
    z_val = float(selected_point.iloc[0][z_marker])

    combined = pd.concat([normal_points[axis_markers], selected_point[axis_markers]], axis=0)
    selected_marker_rows = marker_ranked[
        (marker_ranked["sample_id"] == selected_flagged_id)
        & (marker_ranked["marker"].astype(str).isin(axis_markers))
    ].copy()
    selected_marker_rows = selected_marker_rows.set_index("marker")

    axis_cutoffs: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for marker, axis_name in [(x_marker, "x"), (y_marker, "y"), (z_marker, "z")]:
        if marker not in selected_marker_rows.index:
            continue
        lo_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "lower_cutoff"]]), errors="coerce").iloc[0]
        hi_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "upper_cutoff"]]), errors="coerce").iloc[0]
        if pd.notna(lo_cutoff):
            axis_cutoffs[axis_name].append(float(lo_cutoff))
        if pd.notna(hi_cutoff):
            axis_cutoffs[axis_name].append(float(hi_cutoff))

    def _axis_range(series: pd.Series, extra_values: list[float] | None = None) -> tuple[float, float]:
        numeric = pd.to_numeric(series, errors="coerce").dropna().tolist()
        if extra_values:
            numeric.extend([float(v) for v in extra_values if pd.notna(v)])
        numeric = pd.Series(numeric, dtype=float).dropna()
        if numeric.empty:
            return (0.0, 1.0)
        lo = float(numeric.min())
        hi = float(numeric.max())
        span = hi - lo
        pad = 1.0 if span == 0 else span * 0.15
        return (lo - pad, hi + pad)

    x_range = _axis_range(combined[x_marker], axis_cutoffs["x"])
    y_range = _axis_range(combined[y_marker], axis_cutoffs["y"])
    z_range = _axis_range(combined[z_marker], axis_cutoffs["z"])
    axis_cutoff_colors = {"x": "#FB8C00", "y": "#1E88E5", "z": "#43A047"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(10.8, 7.6), facecolor="#ffffff")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=18, azim=-58)
    ax.set_facecolor("#ffffff")

    if not normal_points.empty:
        ax.scatter(
            normal_points[x_marker].astype(float),
            normal_points[y_marker].astype(float),
            normal_points[z_marker].astype(float),
            c="#8FA3B8",
            alpha=0.38,
            s=30,
            edgecolors="#ffffff",
            linewidths=0.25,
            label=f"Normal group (n={len(normal_points)})",
        )

    ax.scatter([x_val], [y_val], [z_val], c="#E53935", s=300, alpha=0.18, edgecolors="none", zorder=7)
    ax.scatter(
        [x_val],
        [y_val],
        [z_val],
        c="#E53935",
        s=150,
        edgecolors="#ffffff",
        linewidths=1.25,
        label=f"Flagged: {selected_flagged_id}",
        zorder=8,
    )

    for marker, axis_name in [(x_marker, "x"), (y_marker, "y"), (z_marker, "z")]:
        if marker not in selected_marker_rows.index:
            continue
        lo_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "lower_cutoff"]]), errors="coerce").iloc[0]
        hi_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "upper_cutoff"]]), errors="coerce").iloc[0]
        axis_color = axis_cutoff_colors.get(axis_name, "#616161")

        if pd.notna(lo_cutoff):
            _draw_cutoff_line_3d(
                ax=ax,
                axis_name=axis_name,
                cutoff_value=float(lo_cutoff),
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                color=axis_color,
                marker_label=marker,
                cutoff_kind="Lower",
            )
        if pd.notna(hi_cutoff):
            _draw_cutoff_line_3d(
                ax=ax,
                axis_name=axis_name,
                cutoff_value=float(hi_cutoff),
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                color=axis_color,
                marker_label=marker,
                cutoff_kind="Upper",
            )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel(f"X axis: {x_marker}", labelpad=8)
    ax.set_ylabel(f"Y axis: {y_marker}", labelpad=8)
    ax.zaxis.set_rotate_label(True)
    ax.set_zlabel(f"Z axis: {z_marker}", labelpad=16)
    fig.suptitle("Flagged Sample 3D Marker Plot", fontsize=14, fontweight="bold", color="#111111", y=0.985)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.955),
        fontsize=8,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d6dee8",
        framealpha=0.94,
    )
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.07, top=0.90)

    graph_png = BytesIO()
    fig.savefig(graph_png, format="png", dpi=170, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)
    graph_png.seek(0)
    return graph_png


def _build_all_flagged_3d_pngs(
    prioritized_df: pd.DataFrame,
    patient_flagged_df: pd.DataFrame,
    suspected_top_marker_df: pd.DataFrame,
    marker_detail_df: pd.DataFrame,
    disease_to_markers: dict[str, list[str]],
) -> list[tuple[str, BytesIO]]:
    """Build one 3D PNG attachment per flagged sample ID."""
    attachments: list[tuple[str, BytesIO]] = []
    if patient_flagged_df.empty:
        return attachments

    for sample_id in patient_flagged_df["sample_id"].astype(str).tolist():
        one_flagged_df = patient_flagged_df[
            patient_flagged_df["sample_id"].astype(str) == sample_id
        ].head(1)
        png_buffer = _build_default_flagged_3d_png(
            prioritized_df=prioritized_df,
            patient_flagged_df=one_flagged_df,
            suspected_top_marker_df=suspected_top_marker_df,
            marker_detail_df=marker_detail_df,
            disease_to_markers=disease_to_markers,
        )
        if png_buffer is not None:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
            attachments.append((f"3d_flagged_graph_{safe_id}.png", png_buffer))
    return attachments


def build_prediction_results_html_report(
    prioritized_df: pd.DataFrame,
    marker_detail_df: pd.DataFrame,
    disease_to_markers: dict[str, list[str]],
    additional_pattern_df: pd.DataFrame | None = None,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Exclude control/internal samples (LC, HC, IS) from report
    if "is_control" in prioritized_df.columns:
        prioritized_df = prioritized_df[~prioritized_df["is_control"]].copy()

    suspected_ids_df = prioritized_df[prioritized_df["is_flagged"] & ~prioritized_df["is_control"]]
    suspected_ids = ", ".join(suspected_ids_df["sample_id"].astype(str).tolist())

    # Compute additional pattern signal IDs (not in model-flagged set)
    model_flagged_id_set = set(suspected_ids_df["sample_id"].astype(str).tolist())
    if additional_pattern_df is not None and not additional_pattern_df.empty:
        additional_signal_ids = sorted(
            set(additional_pattern_df["sample_id"].astype(str).tolist()) - model_flagged_id_set
        )
    else:
        additional_signal_ids = []

    sections = []
    for _, row in prioritized_df.iterrows():
        sid = str(row.get("sample_id", "-"))
        is_flagged = bool(row.get("is_flagged", False) and not row.get("is_control", False))
        sample_title_class = "flagged-sample-title" if is_flagged else "normal-sample-title"
        row_class = "disease-row" if is_flagged else "normal-row"
        flag_text = " [FLAG]" if is_flagged else ""

        sample_marker_df = marker_detail_df[marker_detail_df["sample_id"].astype(str) == sid].copy()

        diseases = [
            (str(row.get("top_1_disease", "-")), float(row.get("top_1_probability", 0.0))),
            (str(row.get("top_2_disease", "-")), float(row.get("top_2_probability", 0.0))),
            (str(row.get("top_3_disease", "-")), float(row.get("top_3_probability", 0.0))),
        ]

        row_html = []
        for disease_name, prob in diseases:
            disease_markers = _lookup_disease_markers(disease_name, disease_to_markers)
            marker_pick = pick_marker_for_disease(sample_marker_df, disease_markers)
            row_html.append(
                f"""
                <tr class=\"{row_class}\">
                    <td class=\"disease\">{escape(disease_name)}</td>
                    <td class=\"prob\">{prob*100:.2f}%</td>
                    <td>{escape(marker_pick['marker'])}</td>
                    <td>{escape(marker_pick['value'])}</td>
                    <td>{escape(marker_pick['mom'])}</td>
                    <td>{escape(marker_pick['lower_cutoff'])}</td>
                    <td>{escape(marker_pick['upper_cutoff'])}</td>
                    <td>{escape(marker_pick['cutoff_status'])}</td>
                </tr>
                """
            )

        extra_html = ""
        if additional_pattern_df is not None and not additional_pattern_df.empty:
            sample_extra_df = additional_pattern_df[additional_pattern_df["sample_id"].astype(str) == sid].copy()
            if not sample_extra_df.empty:
                extra_rows = []
                for _, extra_row in sample_extra_df.iterrows():
                    extra_rows.append(
                        "<tr>"
                        f"<td>{escape(str(extra_row.get('disease_name', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('pattern_support_percent', '-')))}%</td>"
                        f"<td>{escape(str(extra_row.get('abnormal_marker_count', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('available_pattern_marker_count', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('total_pattern_marker_count', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('abnormal_markers', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('representative_marker', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('value', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('lower_cutoff', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('upper_cutoff', '-')))}</td>"
                        f"<td>{escape(str(extra_row.get('cutoff_status', '-')))}</td>"
                        "</tr>"
                    )

                extra_html = f"""
                <div class=\"extra-pattern-title\">Additional Disease-Pattern Signals</div>
                <table class=\"extra-pattern-table\">
                    <thead>
                        <tr>
                            <th>Disease Pattern</th>
                            <th>Pattern Support %</th>
                            <th>Abnormal Markers</th>
                            <th>Available Pattern Markers</th>
                            <th>Total Pattern Markers</th>
                            <th>Exceeded Marker Names</th>
                            <th>Representative Marker</th>
                            <th>Value</th>
                            <th>Lower Cut-off</th>
                            <th>Upper Cut-off</th>
                            <th>Cut-off Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(extra_rows)}
                    </tbody>
                </table>
                """

        sections.append(
            f"""
            <section class=\"sample-section\">
                <div class=\"sample-title {sample_title_class}\">Sample ID: {escape(sid)}{flag_text}</div>
                <table>
                    <thead>
                        <tr>
                            <th>Disease Pattern</th>
                            <th>Probability</th>
                            <th>Flagged Marker</th>
                            <th>Value</th>
                            <th>MoM</th>
                            <th>Lower Cut-off</th>
                            <th>Upper Cut-off</th>
                            <th>Cut-off Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(row_html)}
                    </tbody>
                </table>
                {extra_html}
            </section>
            """
        )

    has_any_alert = bool(suspected_ids or additional_signal_ids)
    summary_class = "suspect-summary" if has_any_alert else "suspect-summary normal-summary"
    additional_ids_str = ", ".join(escape(sid) for sid in additional_signal_ids)
    if suspected_ids and additional_signal_ids:
        summary_text = (
            f"<strong>Model-flagged sample IDs:</strong> {escape(suspected_ids)}"
            f"<br><strong>Additional pattern signal IDs:</strong> {additional_ids_str}"
        )
    elif suspected_ids:
        summary_text = f"<strong>Suspected disease sample IDs (grouped):</strong> {escape(suspected_ids)}"
    elif additional_signal_ids:
        summary_text = (
            f"<strong>No model-flagged disease samples found.</strong>"
            f"<br><strong>Additional pattern signal IDs:</strong> {additional_ids_str}"
        )
    else:
        summary_text = "<strong>No suspected disease samples found.</strong>"

    csv_b64 = base64.b64encode(prioritized_df.to_csv(index=False).encode("utf-8")).decode("ascii")

    return f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <title>Disease Pattern Detection Report</title>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 24px 32px 56px;
                color: #222;
                background: #fff;
            }}
            .page-title {{
                background: #1e8f4e;
                color: #fff;
                text-align: center;
                font-size: 22px;
                font-weight: 700;
                padding: 12px 16px;
                border-radius: 2px;
                margin-bottom: 18px;
            }}
            .report-note {{
                font-size: 12px;
                color: #666;
                margin-bottom: 18px;
            }}
            .sample-section {{
                margin-bottom: 22px;
                page-break-inside: avoid;
            }}
            .sample-title {{
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            .flagged-sample-title {{ color: #b00020; }}
            .normal-sample-title {{ color: #0b7a3f; }}
            .suspect-summary {{
                margin-bottom: 14px;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid #e59db0;
                background: #fff2f6;
                color: #8f1638;
                font-size: 13px;
            }}
            .suspect-summary.normal-summary {{
                border-color: #a6dcb7;
                background: #eefaf1;
                color: #18633a;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }}
            thead th {{
                background: #1e8f4e;
                color: #fff;
                border: 2px solid #16673a;
                padding: 8px 10px;
                font-size: 14px;
            }}
            tbody td {{
                border: 1px solid #cde8d5;
                padding: 8px 10px;
                font-size: 14px;
                vertical-align: top;
                word-break: break-word;
            }}
            tbody tr.disease-row td {{ color: #b00020; }}
            tbody tr.normal-row td {{ color: #0b7a3f; }}
            tbody tr:nth-child(1) td {{
                font-weight: 700;
                background: #f2fbf5;
            }}
            .disease {{ width: 35%; }}
            .prob {{ width: 12%; text-align: center; }}
            .extra-pattern-title {{
                margin-top: 10px;
                margin-bottom: 6px;
                font-size: 14px;
                font-weight: 700;
                color: #7c5a00;
            }}
            .extra-pattern-table thead th {{
                background: #9a6a00;
                border-color: #7c5500;
            }}
            .extra-pattern-table tbody td {{
                color: #6a4a00;
                border-color: #ecdcb2;
                background: #fff9eb;
            }}
            .footer {{
                margin-top: 40px;
                text-align: center;
                color: #8a8a8a;
                font-size: 12px;
                font-style: italic;
            }}
            @media print {{
                body {{ margin: 16px 18px 32px; }}
                .sample-section {{ page-break-inside: avoid; }}
            }}
            .save-csv-btn {{
                display: inline-block;
                margin-bottom: 16px;
                padding: 8px 18px;
                background: #1e8f4e;
                color: #fff;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                cursor: pointer;
                font-family: Arial, Helvetica, sans-serif;
            }}
            .save-csv-btn:hover {{ background: #16673a; }}
        </style>
    </head>
    <body>
        <div class="page-title">Disease Pattern Detection Report</div>
        <div class="report-note">
            Report uses model top-3 predicted diseases per sample. Added clinical triage fields: MoM, lower/upper cut-off, and cut-off status.
        </div>
        <div class="{summary_class}">{summary_text}</div>
        <button class="save-csv-btn" onclick="downloadReportCSV()">&#8659; Save as CSV</button>
        {''.join(sections)}
        <div class="footer">Generated: {generated_at}</div>
        <script>
            function downloadReportCSV() {{
                var b64 = "{csv_b64}";
                var uri = "data:text/csv;charset=utf-8;base64," + b64;
                var a = document.createElement("a");
                a.href = uri;
                a.download = "prediction_report.csv";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}
        </script>
    </body>
    </html>
    """

st.set_page_config(page_title="IEM Predictor", layout="wide")
st.title("IEM Disease Prediction Web App")
st.caption("Model: best_original_intensive_v1_grid_lr.joblib | Notebook reference: Top6IEM_01.4.10 [Remake](final)")

with st.sidebar:
    st.header("Configuration")
    model_path = Path(
        st.text_input("Model path", value=os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    )
    class_mapping_path = Path(
        st.text_input("Class mapping path", value=os.getenv("CLASS_MAPPING_PATH", str(DEFAULT_CLASS_MAPPING_PATH)))
    )
    train_csv_path = Path(
        st.text_input("Training CSV path", value=os.getenv("TRAIN_CSV_PATH", str(DEFAULT_TRAIN_CSV)))
    )
    # Fixed Discord webhook URL (hidden from sidebar)
    discord_webhook = "https://discord.com/api/webhooks/1484882722643120252/4s9BPlNiEzggngt_RDPDYQmstHOuGhUgFIO0LeZhq4Wv4902fZ8b3J71f3F_GbeZZ31Z"

uploaded_files = st.file_uploader("Upload CSV (one or more files)", type=["csv"], accept_multiple_files=True)
manual_id_col = st.text_input("ID column (optional)", value="")

run_button = st.button("Run Prediction", type="primary", use_container_width=True)
prediction_cache = st.session_state.get("prediction_cache")
use_cached_results = False
if not run_button and prediction_cache is not None:
    run_button = True
    use_cached_results = True

if run_button:
    if not use_cached_results and not uploaded_files:
        st.error("Please upload at least one CSV file first.")
    elif not use_cached_results and not model_path.exists():
        st.error(f"Model file not found: {model_path}")
    else:
        try:
            if use_cached_results:
                predictions_df = prediction_cache["predictions_df"].copy()
                marker_detail_df = prediction_cache["marker_detail_df"].copy()
                detected_id_col = str(prediction_cache.get("detected_id_col", "sample_id"))
                timestamp = str(
                    prediction_cache.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
                )
            else:
                dfs = [pd.read_csv(f) for f in uploaded_files]
                input_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
                effective_train_csv_path = train_csv_path if train_csv_path.exists() else None
                predictions_df, marker_detail_df, detected_id_col = predict_from_dataframe(
                    input_df=input_df,
                    model_path=model_path,
                    class_mapping_path=class_mapping_path,
                    train_csv_path=effective_train_csv_path,
                    id_column=manual_id_col.strip() or None,
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["prediction_cache"] = {
                    "predictions_df": predictions_df.copy(),
                    "marker_detail_df": marker_detail_df.copy(),
                    "detected_id_col": detected_id_col,
                    "timestamp": timestamp,
                }

            triage_df = predictions_df.copy()
            triage_df["is_flagged"] = ~triage_df["top_1_disease"].astype(str).apply(is_normal_disease_name)
            triage_df["is_control"] = triage_df["sample_id"].astype(str).apply(is_control_or_internal_sample)
            triage_df["priority"] = triage_df["is_flagged"].map({True: "FLAG", False: "NORMAL"})
            triage_df["display_group"] = 2
            triage_df.loc[triage_df["is_control"], "display_group"] = 1
            triage_df.loc[triage_df["is_flagged"] & ~triage_df["is_control"], "display_group"] = 0

            prioritized_df = triage_df.sort_values(
                by=["display_group", "top_1_probability"],
                ascending=[True, False],
                kind="stable",
            ).reset_index(drop=True)

            patient_flagged_df = prioritized_df[prioritized_df["is_flagged"] & ~prioritized_df["is_control"]].copy()
            normal_df = prioritized_df[~prioritized_df["is_flagged"]].copy()
            if use_cached_results:
                st.success(f"Showing saved prediction results. ID column used: {detected_id_col}")
            else:
                st.success(f"Prediction complete. ID column used: {detected_id_col}")

            st.subheader("Priority Triage")

            disease_to_markers = build_disease_to_markers_map()
            additional_pattern_df = build_additional_pattern_support_report(
                marker_detail_df,
                class_mapping_path,
            )
            if not additional_pattern_df.empty:
                additional_pattern_df = additional_pattern_df[
                    ~additional_pattern_df["sample_id"].astype(str).apply(is_control_or_internal_sample)
                ].reset_index(drop=True)

            model_flagged_id_set = set(patient_flagged_df["sample_id"].astype(str).tolist())
            additional_only_ids = sorted(
                set(additional_pattern_df["sample_id"].astype(str).tolist()) - model_flagged_id_set
            ) if not additional_pattern_df.empty else []

            st.markdown(f"""
            <div style="display:flex; gap:1rem; margin-bottom:1rem;">
                <div style="flex:1; background:#f0f2f6; border-radius:8px; padding:1rem 1.2rem; text-align:center;">
                    <div style="font-size:0.82rem; color:#555; margin-bottom:4px;">Total Samples</div>
                    <div style="font-size:2rem; font-weight:700; color:#222;">{len(prioritized_df)}</div>
                </div>
                <div style="flex:1; background:#FFEBEB; border-radius:8px; padding:1rem 1.2rem; text-align:center; border-top:4px solid #D32F2F;">
                    <div style="font-size:0.82rem; color:#D32F2F; margin-bottom:4px;">&#9888; Flagged Patients</div>
                    <div style="font-size:2rem; font-weight:700; color:#D32F2F;">{len(patient_flagged_df)}</div>
                </div>
                <div style="flex:1; background:#FFF8E1; border-radius:8px; padding:1rem 1.2rem; text-align:center; border-top:4px solid #F9A825;">
                    <div style="font-size:0.82rem; color:#E65100; margin-bottom:4px;">&#9656; Additional Pattern Signals</div>
                    <div style="font-size:2rem; font-weight:700; color:#F9A825;">{len(additional_only_ids)}</div>
                </div>
                <div style="flex:1; background:#E8F5E9; border-radius:8px; padding:1rem 1.2rem; text-align:center; border-top:4px solid #2E7D32;">
                    <div style="font-size:0.82rem; color:#2E7D32; margin-bottom:4px;">&#10003; Normal / Other</div>
                    <div style="font-size:2rem; font-weight:700; color:#2E7D32;">{len(normal_df)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not patient_flagged_df.empty:
                flagged_ids = ", ".join(patient_flagged_df["sample_id"].astype(str).tolist())
                if additional_only_ids:
                    add_ids_str = ", ".join(additional_only_ids)
                    st.error(
                        f"Model-flagged IDs: {flagged_ids}  |  Additional pattern signal IDs: {add_ids_str}"
                    )
                else:
                    st.error(f"Priority sample IDs for doctor review: {flagged_ids}")
            elif additional_only_ids:
                add_ids_str = ", ".join(additional_only_ids)
                st.warning(f"No model-flagged samples. Additional pattern signal IDs: {add_ids_str}")
            else:
                st.success("No flagged disease samples detected in top-1 prediction.")
            top_marker_df = pick_top_marker_per_sample(
                marker_detail_df,
                prioritized_df=prioritized_df,
                disease_to_markers=disease_to_markers,
            )
            suspected_top_marker_df = (
                patient_flagged_df.merge(top_marker_df, on="sample_id", how="left")
                .sort_values(by=["sample_id"], kind="stable")
                .reset_index(drop=True)
            )
            suspected_html_report = build_suspected_case_html_report(suspected_top_marker_df)
            prediction_html_report = build_prediction_results_html_report(
                prioritized_df,
                marker_detail_df,
                disease_to_markers,
                additional_pattern_df=additional_pattern_df,
            )

            tab1, tab2, tab3, tab4 = st.tabs([
                "Prediction Results",
                "MoM and Cut-off Details",
                "Additional Pattern Diseases",
                "3D Flagged Graph",
            ])

            with tab1:
                show_df = prioritized_df.copy()
                for c in ["top_1_probability", "top_2_probability", "top_3_probability"]:
                    show_df[c] = (show_df[c] * 100).round(2)
                show_df = show_df.rename(
                    columns={
                        "top_1_probability": "top_1_probability_percent",
                        "top_2_probability": "top_2_probability_percent",
                        "top_3_probability": "top_3_probability_percent",
                    }
                )
                show_cols = [
                    "sample_id",
                    "priority",
                    "top_1_disease",
                    "top_1_probability_percent",
                    "top_2_disease",
                    "top_2_probability_percent",
                    "top_3_disease",
                    "top_3_probability_percent",
                ]
                show_df = show_df[[c for c in show_cols if c in show_df.columns]]
                st.dataframe(show_df, use_container_width=True)
                st.download_button(
                    label="Download prediction CSV",
                    data=prioritized_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{timestamp}.csv",
                    mime="text/csv",
                )

                st.subheader("Prediction HTML Report")
                components.html(prediction_html_report, height=700, scrolling=True)
                st.download_button(
                    label="Download Prediction HTML Report",
                    data=prediction_html_report.encode("utf-8"),
                    file_name=f"inference_prediction_report_{timestamp}.html",
                    mime="text/html",
                )

            with tab2:
                st.caption("Only suspected disease cases and only one top important marker per sample.")
                if suspected_top_marker_df.empty:
                    st.info("No suspected disease cases found.")
                else:
                    display_cols = [
                        "sample_id",
                        "top_1_disease",
                        "top_1_probability",
                        "marker",
                        "value",
                        "MoM",
                        "lower_cutoff",
                        "upper_cutoff",
                        "top_2_disease",
                        "top_2_probability",
                        "top_3_disease",
                        "top_3_probability",
                    ]
                    show_suspected = suspected_top_marker_df[[c for c in display_cols if c in suspected_top_marker_df.columns]].copy()
                    for col in ["top_1_probability", "top_2_probability", "top_3_probability"]:
                        if col in show_suspected.columns:
                            show_suspected[col] = (pd.to_numeric(show_suspected[col], errors="coerce") * 100).round(2)
                    for col in ["value", "MoM", "lower_cutoff", "upper_cutoff"]:
                        if col in show_suspected.columns:
                            show_suspected[col] = pd.to_numeric(show_suspected[col], errors="coerce").round(4)
                    if "MoM" in show_suspected.columns:
                        show_suspected["MoM"] = pd.to_numeric(show_suspected["MoM"], errors="coerce").round(2)
                    show_suspected = show_suspected.rename(
                        columns={
                            "top_1_probability": "top_1_probability_percent",
                            "top_2_probability": "top_2_probability_percent",
                            "top_3_probability": "top_3_probability_percent",
                            "marker": "top_important_marker",
                        }
                    )
                    st.dataframe(show_suspected, use_container_width=True)

                st.download_button(
                    label="Download Suspected Top-Marker MoM/Cut-off CSV",
                    data=suspected_top_marker_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"suspected_top_marker_mom_cutoff_{timestamp}.csv",
                    mime="text/csv",
                )

            with tab3:
                st.caption("Diseases outside the model training set, reported when more than 30% of the pattern markers found in that sample are outside cut-off.")
                if additional_pattern_df.empty:
                    st.info("No additional disease-pattern signals exceeded the >30% abnormal-marker threshold among markers available in the sample.")
                else:
                    st.dataframe(additional_pattern_df, use_container_width=True)
                st.download_button(
                    label="Download Additional Pattern Disease CSV",
                    data=additional_pattern_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"additional_pattern_diseases_{timestamp}.csv",
                    mime="text/csv",
                )

            with tab4:
                st.caption("3D view for flagged samples: main marker + ratio markers related to the main marker, compared against normal group with axis-specific cut-off lines.")

                if patient_flagged_df.empty:
                    st.info("No flagged samples available for 3D visualization.")
                else:
                    flagged_ids = patient_flagged_df["sample_id"].astype(str).tolist()
                    selected_flagged_id = st.selectbox(
                        "Select flagged sample ID",
                        flagged_ids,
                        key="flagged_3d_sample_id",
                    )

                    selected_row_df = prioritized_df[prioritized_df["sample_id"].astype(str) == selected_flagged_id]
                    selected_disease = str(selected_row_df.iloc[0]["top_1_disease"]) if not selected_row_df.empty else "-"
                    selected_top_marker_df = suspected_top_marker_df[
                        suspected_top_marker_df["sample_id"].astype(str) == selected_flagged_id
                    ]
                    selected_main_marker = (
                        str(selected_top_marker_df.iloc[0].get("marker", "")).strip()
                        if not selected_top_marker_df.empty
                        else ""
                    )

                    available_markers = sorted(set(marker_detail_df["marker"].dropna().astype(str)))
                    axis_markers = _build_flagged_3d_axis_markers(
                        main_marker=selected_main_marker,
                        suspected_disease=selected_disease,
                        disease_to_markers=disease_to_markers,
                        available_markers=available_markers,
                    )

                    if len(axis_markers) != 3:
                        st.warning(
                            "Cannot build 3D axes from the selected flagged case. "
                            "Need one main marker and at least two related markers available in the data."
                        )
                    else:
                        work = _prepare_marker_candidates(marker_detail_df)
                        keep_cols = ["sample_id", "marker", "value", "MoM", "lower_cutoff", "upper_cutoff"]
                        work = work[[c for c in keep_cols if c in work.columns]].copy()
                        work = work[work["marker"].astype(str).isin(axis_markers)].copy()
                        work["sample_id"] = work["sample_id"].astype(str)
                        work["value"] = pd.to_numeric(work["value"], errors="coerce")
                        work["MoM"] = pd.to_numeric(work["MoM"], errors="coerce")
                        work["lower_cutoff"] = pd.to_numeric(work["lower_cutoff"], errors="coerce")
                        work["upper_cutoff"] = pd.to_numeric(work["upper_cutoff"], errors="coerce")

                        marker_ranked = work.sort_values(
                            by=["sample_id", "marker", "MoM"],
                            ascending=[True, True, False],
                            kind="stable",
                        ).drop_duplicates(subset=["sample_id", "marker"], keep="first")

                        value_pivot = marker_ranked.pivot(index="sample_id", columns="marker", values="value")
                        mom_pivot = marker_ranked.pivot(index="sample_id", columns="marker", values="MoM")

                        normal_ids = set(
                            prioritized_df.loc[
                                (~prioritized_df["is_flagged"]) & (~prioritized_df["is_control"]),
                                "sample_id",
                            ].astype(str)
                        )

                        normal_points = value_pivot[value_pivot.index.isin(normal_ids)].copy()
                        normal_points = normal_points.dropna(subset=axis_markers, how="any")

                        selected_point = value_pivot[value_pivot.index == selected_flagged_id].copy()
                        selected_point = selected_point.dropna(subset=axis_markers, how="any")

                        if selected_point.empty:
                            st.warning("Selected sample does not have complete values for all 3 axis markers.")
                        else:
                            x_marker, y_marker, z_marker = axis_markers
                            x_val = float(selected_point.iloc[0][x_marker])
                            y_val = float(selected_point.iloc[0][y_marker])
                            z_val = float(selected_point.iloc[0][z_marker])

                            combined = pd.concat(
                                [normal_points[axis_markers], selected_point[axis_markers]],
                                axis=0,
                            )

                            selected_marker_rows = marker_ranked[
                                (marker_ranked["sample_id"] == selected_flagged_id)
                                & (marker_ranked["marker"].astype(str).isin(axis_markers))
                            ].copy()
                            selected_marker_rows = selected_marker_rows.set_index("marker")

                            axis_cutoffs: dict[str, list[float]] = {"x": [], "y": [], "z": []}
                            for marker, axis_name in [(x_marker, "x"), (y_marker, "y"), (z_marker, "z")]:
                                if marker not in selected_marker_rows.index:
                                    continue
                                lo_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "lower_cutoff"]]), errors="coerce").iloc[0]
                                hi_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "upper_cutoff"]]), errors="coerce").iloc[0]
                                if pd.notna(lo_cutoff):
                                    axis_cutoffs[axis_name].append(float(lo_cutoff))
                                if pd.notna(hi_cutoff):
                                    axis_cutoffs[axis_name].append(float(hi_cutoff))

                            def _axis_range(series: pd.Series, extra_values: list[float] | None = None) -> tuple[float, float]:
                                numeric = pd.to_numeric(series, errors="coerce").dropna().tolist()
                                if extra_values:
                                    numeric.extend([float(v) for v in extra_values if pd.notna(v)])
                                numeric = pd.Series(numeric, dtype=float).dropna()
                                if numeric.empty:
                                    return (0.0, 1.0)
                                lo = float(numeric.min())
                                hi = float(numeric.max())
                                span = hi - lo
                                pad = 1.0 if span == 0 else span * 0.15
                                return (lo - pad, hi + pad)

                            x_range = _axis_range(combined[x_marker], axis_cutoffs["x"])
                            y_range = _axis_range(combined[y_marker], axis_cutoffs["y"])
                            z_range = _axis_range(combined[z_marker], axis_cutoffs["z"])
                            axis_cutoff_colors = {
                                "x": "#FB8C00",  # orange
                                "y": "#1E88E5",  # blue
                                "z": "#43A047",  # green
                            }

                            details = []
                            for marker in axis_markers:
                                m_val = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "value"]]), errors="coerce").iloc[0] if marker in selected_marker_rows.index else np.nan
                                m_mom = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "MoM"]]), errors="coerce").iloc[0] if marker in selected_marker_rows.index else np.nan
                                v_text = "-" if pd.isna(m_val) else f"{float(m_val):.4f}"
                                mom_text = "-" if pd.isna(m_mom) else f"{float(m_mom):.2f}"
                                details.append(f"{marker}: {v_text} (MoM {mom_text})")

                            sample_legend_label = (
                                f"Sample {selected_flagged_id} | {selected_disease}\n"
                                + " | ".join(details)
                            )

                            interactive_3d = st.toggle(
                                "Interactive 3D (drag to rotate)",
                                value=go is not None,
                                key="flagged_3d_interactive",
                                help="If enabled, the graph is rendered with Plotly and can be rotated with mouse drag.",
                            )
                            if interactive_3d and go is None:
                                st.info("Interactive mode requires plotly. Falling back to static mode.")
                                interactive_3d = False

                            if interactive_3d:
                                max_normal_points = st.slider(
                                    "Max normal points (performance)",
                                    min_value=100,
                                    max_value=5000,
                                    value=1500,
                                    step=100,
                                    key="flagged_3d_max_normal_points",
                                )
                                normal_plot_points = normal_points.copy()
                                if len(normal_plot_points) > max_normal_points:
                                    normal_plot_points = normal_plot_points.sample(max_normal_points, random_state=42)

                                fig3d = go.Figure()
                                if not normal_plot_points.empty:
                                    fig3d.add_trace(
                                        go.Scatter3d(
                                            x=normal_plot_points[x_marker].astype(float),
                                            y=normal_plot_points[y_marker].astype(float),
                                            z=normal_plot_points[z_marker].astype(float),
                                            mode="markers",
                                            marker=dict(size=4.0, color="#8FA3B8", opacity=0.34),
                                            text=normal_plot_points.index.astype(str),
                                            hovertemplate=(
                                                "Sample: %{text}<br>"
                                                + f"{x_marker}: %{{x:.4f}}<br>"
                                                + f"{y_marker}: %{{y:.4f}}<br>"
                                                + f"{z_marker}: %{{z:.4f}}"
                                                + "<extra>Normal</extra>"
                                            ),
                                            name=f"Normal group (shown={len(normal_plot_points)})",
                                        )
                                    )

                                fig3d.add_trace(
                                    go.Scatter3d(
                                        x=[x_val],
                                        y=[y_val],
                                        z=[z_val],
                                        mode="markers",
                                        marker=dict(size=15, color="rgba(229,57,53,0.20)"),
                                        hoverinfo="skip",
                                        showlegend=False,
                                    )
                                )

                                fig3d.add_trace(
                                    go.Scatter3d(
                                        x=[x_val],
                                        y=[y_val],
                                        z=[z_val],
                                        mode="markers",
                                        marker=dict(size=8.8, color="#E53935", line=dict(color="#ffffff", width=1.2)),
                                        hovertemplate=(
                                            f"Sample: {selected_flagged_id}<br>"
                                            + f"{x_marker}: %{{x:.4f}}<br>"
                                            + f"{y_marker}: %{{y:.4f}}<br>"
                                            + f"{z_marker}: %{{z:.4f}}"
                                            + "<extra>Flagged</extra>"
                                        ),
                                        name=sample_legend_label,
                                    )
                                )

                                for marker, axis_name in [(x_marker, "x"), (y_marker, "y"), (z_marker, "z")]:
                                    if marker not in selected_marker_rows.index:
                                        continue
                                    lo_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "lower_cutoff"]]), errors="coerce").iloc[0]
                                    hi_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "upper_cutoff"]]), errors="coerce").iloc[0]
                                    axis_color = axis_cutoff_colors.get(axis_name, "#616161")

                                    if pd.notna(lo_cutoff):
                                        line_x, line_y, line_z = _build_plotly_cutoff_line(
                                            axis_name=axis_name,
                                            cutoff_value=float(lo_cutoff),
                                            x_range=x_range,
                                            y_range=y_range,
                                            z_range=z_range,
                                        )
                                        fig3d.add_trace(
                                            go.Scatter3d(
                                                x=line_x,
                                                y=line_y,
                                                z=line_z,
                                                mode="lines",
                                                line=dict(color=axis_color, width=2.5, dash="dash"),
                                                name=f"{marker} lower cut-off",
                                                showlegend=False,
                                            )
                                        )
                                    if pd.notna(hi_cutoff):
                                        line_x, line_y, line_z = _build_plotly_cutoff_line(
                                            axis_name=axis_name,
                                            cutoff_value=float(hi_cutoff),
                                            x_range=x_range,
                                            y_range=y_range,
                                            z_range=z_range,
                                        )
                                        fig3d.add_trace(
                                            go.Scatter3d(
                                                x=line_x,
                                                y=line_y,
                                                z=line_z,
                                                mode="lines",
                                                line=dict(color=axis_color, width=2.5, dash="dot"),
                                                name=f"{marker} upper cut-off",
                                                showlegend=False,
                                            )
                                        )

                                fig3d.update_layout(
                                    template="plotly_white",
                                    paper_bgcolor="#f6f8fb",
                                    plot_bgcolor="#f6f8fb",
                                    font=dict(color="#111111"),
                                    margin=dict(l=6, r=6, b=56, t=74),
                                    title={
                                        "text": "Flagged Sample 3D Marker Plot",
                                        "x": 0.02,
                                        "xanchor": "left",
                                        "font": {"color": "#111111", "size": 17},
                                    },
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.13,
                                        x=0,
                                        bgcolor="rgba(255,255,255,0.88)",
                                        bordercolor="#dbe3ec",
                                        borderwidth=1,
                                        font=dict(color="#111111"),
                                    ),
                                    scene=dict(
                                        xaxis=dict(
                                            title=dict(text=f"X axis: {x_marker}", font=dict(color="#111111")),
                                            range=list(x_range),
                                            showbackground=True,
                                            backgroundcolor="#fbfcfe",
                                            gridcolor="#d6dee8",
                                            zerolinecolor="#c2ceda",
                                            tickfont=dict(color="#111111"),
                                            showspikes=False,
                                        ),
                                        yaxis=dict(
                                            title=dict(text=f"Y axis: {y_marker}", font=dict(color="#111111")),
                                            range=list(y_range),
                                            showbackground=True,
                                            backgroundcolor="#fbfcfe",
                                            gridcolor="#d6dee8",
                                            zerolinecolor="#c2ceda",
                                            tickfont=dict(color="#111111"),
                                            showspikes=False,
                                        ),
                                        zaxis=dict(
                                            title=dict(text=f"Z axis: {z_marker}", font=dict(color="#111111")),
                                            range=list(z_range),
                                            showbackground=True,
                                            backgroundcolor="#fbfcfe",
                                            gridcolor="#d6dee8",
                                            zerolinecolor="#c2ceda",
                                            tickfont=dict(color="#111111"),
                                            showspikes=False,
                                        ),
                                        aspectmode="manual",
                                        aspectratio=dict(x=1.12, y=1.0, z=0.95),
                                        camera=dict(eye=dict(x=1.46, y=-1.44, z=1.08)),
                                    ),
                                )
                                st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})
                            else:
                                plt.style.use("seaborn-v0_8-whitegrid")
                                fig = plt.figure(figsize=(10.8, 7.6), facecolor="#ffffff")
                                ax = fig.add_subplot(111, projection="3d")
                                ax.view_init(elev=18, azim=-58)
                                ax.set_facecolor("#fbfcfe")
                                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                                    axis.pane.fill = True
                                    axis.pane.set_alpha(0.10)
                                    axis.pane.set_edgecolor("#d6dee8")
                                ax.set_box_aspect((1.1, 1.0, 1.0))

                                if not normal_points.empty:
                                    ax.scatter(
                                        normal_points[x_marker].astype(float),
                                        normal_points[y_marker].astype(float),
                                        normal_points[z_marker].astype(float),
                                        c="#8FA3B8",
                                        alpha=0.38,
                                        s=30,
                                        edgecolors="#ffffff",
                                        linewidths=0.25,
                                        label=f"Normal group (n={len(normal_points)})",
                                    )

                                ax.scatter(
                                    [x_val],
                                    [y_val],
                                    [z_val],
                                    c="#E53935",
                                    s=300,
                                    alpha=0.18,
                                    edgecolors="none",
                                    zorder=7,
                                )

                                ax.scatter(
                                    [x_val],
                                    [y_val],
                                    [z_val],
                                    c="#E53935",
                                    s=150,
                                    edgecolors="#ffffff",
                                    linewidths=1.25,
                                    label=sample_legend_label,
                                    zorder=8,
                                )

                                for marker, axis_name in [(x_marker, "x"), (y_marker, "y"), (z_marker, "z")]:
                                    if marker not in selected_marker_rows.index:
                                        continue
                                    lo_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "lower_cutoff"]]), errors="coerce").iloc[0]
                                    hi_cutoff = pd.to_numeric(pd.Series([selected_marker_rows.at[marker, "upper_cutoff"]]), errors="coerce").iloc[0]
                                    axis_color = axis_cutoff_colors.get(axis_name, "#616161")

                                    if pd.notna(lo_cutoff):
                                        _draw_cutoff_line_3d(
                                            ax=ax,
                                            axis_name=axis_name,
                                            cutoff_value=float(lo_cutoff),
                                            x_range=x_range,
                                            y_range=y_range,
                                            z_range=z_range,
                                            color=axis_color,
                                            marker_label=marker,
                                            cutoff_kind="Lower",
                                        )
                                    if pd.notna(hi_cutoff):
                                        _draw_cutoff_line_3d(
                                            ax=ax,
                                            axis_name=axis_name,
                                            cutoff_value=float(hi_cutoff),
                                            x_range=x_range,
                                            y_range=y_range,
                                            z_range=z_range,
                                            color=axis_color,
                                            marker_label=marker,
                                            cutoff_kind="Upper",
                                        )

                                z_span = z_range[1] - z_range[0]
                                z_pad = 1.0 if z_span == 0 else z_span * 0.12
                                z_range_static = (z_range[0] - z_pad, z_range[1] + z_pad)

                                ax.set_xlim(x_range)
                                ax.set_ylim(y_range)
                                ax.set_zlim(z_range_static)
                                ax.set_xlabel(f"X axis: {x_marker}", labelpad=8)
                                ax.set_ylabel(f"Y axis: {y_marker}", labelpad=8)
                                ax.zaxis.set_rotate_label(True)
                                ax.set_zlabel(f"Z axis: {z_marker}", labelpad=20)
                                fig.suptitle(
                                    "Flagged Sample 3D Marker Plot",
                                    fontsize=14,
                                    fontweight="bold",
                                    color="#ffffff",
                                    y=0.985,
                                )
                                ax.xaxis.label.set_color("#111111")
                                ax.yaxis.label.set_color("#111111")
                                ax.zaxis.label.set_color("#111111")
                                ax.tick_params(axis="x", which="major", labelsize=9, colors="#111111")
                                ax.tick_params(axis="y", which="major", labelsize=9, colors="#111111")
                                ax.tick_params(axis="z", which="major", labelsize=9, colors="#111111")
                                ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.38, color="#afbccb")
                                ax.legend(
                                    loc="upper left",
                                    bbox_to_anchor=(0.01, 0.955),
                                    fontsize=8,
                                    frameon=True,
                                    facecolor="#ffffff",
                                    edgecolor="#d6dee8",
                                    framealpha=0.94,
                                    labelcolor="#111111",
                                )
                                fig.subplots_adjust(left=0.06, right=0.86, bottom=0.07, top=0.90)

                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)

                            st.caption(
                                f"Axes: X axis={x_marker}, Y axis={y_marker}, Z axis={z_marker}. "
                                "Cut-off line colors by axis: X=Orange, Y=Blue, Z=Green. "
                                "Lower cut-off uses dashed line, upper cut-off uses dotted line. "
                                "All axis ranges are expanded to cover all cut-off values."
                            )

            if (not use_cached_results) and discord_webhook.strip():
                try:
                    import importlib
                    import discord_notifier as discord_notifier_module

                    discord_notifier_module = importlib.reload(discord_notifier_module)
                    send_iem_screening_report = getattr(
                        discord_notifier_module,
                        "send_iem_screening_report",
                        None,
                    )
                    if send_iem_screening_report is None:
                        raise RuntimeError(
                            "discord_notifier.send_iem_screening_report not found in loaded module"
                        )

                    flagged_ids_list = (
                        patient_flagged_df["sample_id"].astype(str).tolist()
                        if not patient_flagged_df.empty
                        else []
                    )
                    additional_signal_ids_list = additional_only_ids if additional_only_ids else []

                    top1_for_discord = suspected_top_marker_df[[
                        "sample_id", "top_1_disease", "top_1_probability", "marker", "value", "MoM", "lower_cutoff", "upper_cutoff"
                    ]].copy() if not suspected_top_marker_df.empty else pd.DataFrame()

                    additional_for_discord = pd.DataFrame()
                    if not additional_pattern_df.empty and additional_signal_ids_list:
                        additional_for_discord = additional_pattern_df[
                            additional_pattern_df["sample_id"].astype(str).isin(additional_signal_ids_list)
                        ].copy()
                        keep_cols = [
                            "sample_id",
                            "disease_name",
                            "pattern_support_percent",
                            "representative_marker",
                            "value",
                            "MoM",
                            "lower_cutoff",
                            "upper_cutoff",
                        ]
                        additional_for_discord = additional_for_discord[[
                            c for c in keep_cols if c in additional_for_discord.columns
                        ]]
                        additional_for_discord = additional_for_discord.rename(
                            columns={
                                "sample_id": "Sample ID",
                                "disease_name": "Pattern",
                                "pattern_support_percent": "Support_%",
                                "representative_marker": "Marker",
                                "value": "Value",
                                "MoM": "MoM",
                                "lower_cutoff": "Lower_Cutoff",
                                "upper_cutoff": "Upper_Cutoff",
                            }
                        )

                    if not top1_for_discord.empty:
                        top1_for_discord["top_1_probability"] = (
                            top1_for_discord["top_1_probability"] * 100
                        ).round(2)
                        top1_for_discord = top1_for_discord.rename(
                            columns={
                                "top_1_probability": "top_1_prob_%",
                                "marker": "Marker",
                                "value": "Value",
                                "lower_cutoff": "Lower_Cutoff",
                                "upper_cutoff": "Upper_Cutoff",
                            }
                        )

                    send_iem_screening_report(
                        webhook_url=discord_webhook.strip(),
                        total_samples=len(prioritized_df),
                        normal_count=len(normal_df),
                        flagged_count=len(patient_flagged_df),
                        additional_signal_count=len(additional_only_ids),
                        flagged_ids=flagged_ids_list,
                        additional_signal_ids=additional_signal_ids_list,
                        additional_signal_details_df=additional_for_discord,
                        top1_predictions_df=top1_for_discord,
                        graph_images=_build_all_flagged_3d_pngs(
                            prioritized_df=prioritized_df,
                            patient_flagged_df=patient_flagged_df,
                            suspected_top_marker_df=suspected_top_marker_df,
                            marker_detail_df=marker_detail_df,
                            disease_to_markers=disease_to_markers,
                        ),
                    )
                    st.info("Result sent to Discord automatically (PNG report).")
                except Exception as auto_send_exc:
                    st.warning(f"Auto-send to Discord failed: {auto_send_exc}")

        except Exception as exc:
            st.exception(exc)
