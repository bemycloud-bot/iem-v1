from pathlib import Path
import warnings

import pandas as pd

import inference
import streamlit_app as app


def main() -> None:
    warnings.simplefilter("ignore")
    src = pd.read_csv("../new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv")
    _, marker_detail_df, _ = inference.predict_from_dataframe(
        src,
        model_path=Path("../comparisons/best_original_intensive_v1_grid_lr.joblib"),
        class_mapping_path=Path("../comparisons/class_mapping.csv"),
        train_csv_path=Path("../new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv"),
    )
    additional = app.build_additional_pattern_support_report(
        marker_detail_df,
        Path("../comparisons/class_mapping.csv"),
    )

    target_samples = ["256901020048", "256901040036", "HC763951", "LC763951", "IS"]
    wanted_markers = [
        "MET",
        "MET/CIT",
        "MET/PHE",
        "MET/TYR",
        "ORN/CIT",
        "PHE/TYR",
        "TYR",
        "CIT/PHE",
        "CIT/TYR",
        "ARG/PHE",
        "ARG/ORN",
    ]

    subset = marker_detail_df[
        marker_detail_df["sample_id"].astype(str).isin(target_samples)
        & marker_detail_df["marker"].isin(wanted_markers)
    ][["sample_id", "marker", "value", "below_cutoff", "above_cutoff"]].sort_values(["sample_id", "marker"])

    target_additional = additional[
        additional["sample_id"].astype(str).isin(target_samples)
    ][[
        "sample_id",
        "disease_name",
        "pattern_support_percent",
        "abnormal_marker_count",
        "available_pattern_marker_count",
        "total_pattern_marker_count",
        "available_pattern_markers",
        "abnormal_markers",
    ]].sort_values(["sample_id", "pattern_support_percent", "disease_name"], ascending=[True, False, True])

    print(f"MARKER_ROWS={len(marker_detail_df)}")
    print(f"ADDITIONAL_ROWS={len(additional)}")
    print("MARKERS_START")
    print(subset.to_csv(index=False))
    print("ADDITIONAL_START")
    print(target_additional.to_csv(index=False))


if __name__ == "__main__":
    main()
