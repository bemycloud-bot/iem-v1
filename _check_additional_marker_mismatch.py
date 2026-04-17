from pathlib import Path
import re

import pandas as pd

import inference
import streamlit_app as app


def main() -> None:
    src = pd.read_csv("../new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv")
    _, marker_detail_df, _ = inference.predict_from_dataframe(
        input_df=src,
        model_path=Path("../comparisons/best_original_intensive_v1_grid_lr.joblib"),
        class_mapping_path=Path("../comparisons/class_mapping.csv"),
        train_csv_path=Path("../new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv"),
    )

    additional = app.build_additional_pattern_support_report(
        marker_detail_df, Path("../comparisons/class_mapping.csv")
    )
    if not additional.empty:
        additional = additional[
            ~additional["sample_id"].astype(str).apply(app.is_control_or_internal_sample)
        ].reset_index(drop=True)

    assets = app.load_feature_engineering_assets()
    pattern_df = assets.pattern_df.copy()
    trained = app.load_trained_disease_names(Path("../comparisons/class_mapping.csv"))
    trained_canonical = {app._canonicalize_disease_name(name) for name in trained}

    disease_map = {}
    for raw_name, group in pattern_df.dropna(subset=["DiseaseName"]).groupby("DiseaseName"):
        markers = sorted(set(group["xml_name"].dropna().astype(str)))
        if not markers:
            continue
        mapped_names = (
            group["DiseaseName_mapped"].dropna().astype(str).tolist()
            if "DiseaseName_mapped" in group.columns
            else []
        )
        mapped_name = mapped_names[0] if mapped_names else str(raw_name)
        raw_aliases = app._expand_disease_aliases(str(raw_name))
        mapped_aliases = app._expand_disease_aliases(mapped_name)
        if raw_aliases & trained_canonical or mapped_aliases & trained_canonical:
            continue
        display_name = re.sub(r"\s*\(\d+\)\s*$", "", str(raw_name)).strip()
        disease_map[display_name] = markers

    prepared = app._prepare_marker_candidates(marker_detail_df)
    rows = []
    for _, r in additional.iterrows():
        sid = str(r["sample_id"])
        dname = str(r["disease_name"])
        pattern_markers = set(disease_map.get(dname, []))
        sample_markers = set(
            prepared.loc[prepared["sample_id"].astype(str) == sid, "marker"].astype(str)
        )
        missing = sorted(pattern_markers - sample_markers)
        if missing:
            rows.append(
                {
                    "sample_id": sid,
                    "disease_name": dname,
                    "missing_count": len(missing),
                    "total_pattern_count": len(pattern_markers),
                    "missing_markers": ", ".join(missing[:20]),
                }
            )

    res = pd.DataFrame(rows)
    print(f"ADDITIONAL_ROWS={len(additional)}")
    print(f"ROWS_WITH_MISSING_PATTERN_MARKERS={len(res)}")
    if res.empty:
        print("NO_MISSING_MARKERS_FOR_ADDITIONAL_ROWS")
        return

    res = res.sort_values(
        ["missing_count", "sample_id", "disease_name"],
        ascending=[False, True, True],
        kind="stable",
    )
    print("TOP_MISSING_START")
    print(res.head(50).to_csv(index=False))
    print("TOP_MISSING_END")


if __name__ == "__main__":
    main()
