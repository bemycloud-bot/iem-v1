from __future__ import annotations

from io import BytesIO
from typing import Optional
from datetime import datetime
import textwrap

import pandas as pd
import requests
import matplotlib.pyplot as plt


def send_discord_result(
    webhook_url: str,
    summary_text: str,
    predictions_df: Optional[pd.DataFrame] = None,
    timeout: int = 20,
) -> None:
    if not webhook_url:
        return

    response = requests.post(webhook_url, json={"content": summary_text}, timeout=timeout)
    response.raise_for_status()

    if predictions_df is not None and not predictions_df.empty:
        buff = BytesIO()
        predictions_df.to_csv(buff, index=False)
        buff.seek(0)
        files = {"file": ("predictions.csv", buff.getvalue(), "text/csv")}
        payload = {"content": "Prediction CSV attached."}
        response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
        response.raise_for_status()


def send_iem_screening_report(
    webhook_url: str,
    total_samples: int,
    normal_count: int,
    flagged_count: int,
    additional_signal_count: int,
    flagged_ids: list,
    additional_signal_ids: list,
    additional_signal_details_df: Optional[pd.DataFrame] = None,
    top1_predictions_df: Optional[pd.DataFrame] = None,
    graph_image_path: Optional[str | BytesIO] = None,
    graph_images: Optional[list[tuple[str, BytesIO]]] = None,
    timeout: int = 20,
) -> None:
    """Send IEM screening report to Discord with summary + PNG report."""
    if not webhook_url:
        return

    # Build summary text
    summary_lines = [
        "**IEM Screening Summary**",
        f"Total samples processed: **{total_samples}**",
        f"Normal / no signal: **{normal_count}**",
        (
            "Cases requiring attention: "
            f"**{flagged_count}** model-flagged, **{additional_signal_count}** additional pattern signals"
        ),
        "",
        "**Model-flagged sample IDs**",
        ", ".join(map(str, flagged_ids)) if flagged_ids else "None",
    ]
    if additional_signal_ids:
        summary_lines.extend([
            "",
            "**Additional pattern-signal sample IDs**",
            ", ".join(map(str, additional_signal_ids)),
        ])
    
    summary_text = "\n".join(summary_lines)
    
    # Send initial message with summary
    embed = {
        "title": "IEM Screening Report",
        "color": 16711680 if flagged_count > 0 else 65280,  # Red if flagged, green otherwise
        "fields": [
            {"name": "Total Samples", "value": str(total_samples), "inline": True},
            {"name": "Normal / No Signal", "value": str(normal_count), "inline": True},
            {"name": "Model-Flagged Cases", "value": str(flagged_count), "inline": True},
            {"name": "Additional Pattern Signals", "value": str(additional_signal_count), "inline": True},
            {
                "name": "Model-Flagged Sample IDs",
                "value": ", ".join(map(str, flagged_ids)) if flagged_ids else "None",
                "inline": False,
            },
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if additional_signal_ids:
        embed["fields"].append({
            "name": "Additional Pattern-Signal Sample IDs",
            "value": ", ".join(map(str, additional_signal_ids))[:1024],
            "inline": False,
        })
    
    payload = {
        "content": summary_text,
        "embeds": [embed],
    }
    
    response = requests.post(webhook_url, json=payload, timeout=timeout)
    response.raise_for_status()
    
    # Build and send PNG summary report (replaces CSV attachment)
    report_png = build_iem_screening_png_report(
        total_samples=total_samples,
        normal_count=normal_count,
        flagged_count=flagged_count,
        additional_signal_count=additional_signal_count,
        flagged_ids=flagged_ids,
        additional_signal_ids=additional_signal_ids,
        top1_predictions_df=top1_predictions_df,
    )
    report_png.seek(0)
    files = {"file": ("iem_screening_report.png", report_png.getvalue(), "image/png")}
    payload = {"content": "Attached: Formal IEM screening report (PNG)."}
    response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
    response.raise_for_status()

    # Build and send PNG for additional pattern-signal sample IDs
    additional_ids_png = build_additional_signal_ids_png(
        additional_signal_ids=additional_signal_ids,
        additional_signal_details_df=additional_signal_details_df,
    )
    additional_ids_png.seek(0)
    files = {"file": ("additional_pattern_signal_ids.png", additional_ids_png.getvalue(), "image/png")}
    payload = {"content": "Attached: Additional pattern-signal sample IDs (PNG)."}
    response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
    response.raise_for_status()
    
    # Send 3D graph image
    if graph_image_path and isinstance(graph_image_path, str):
        try:
            with open(graph_image_path, "rb") as f:
                files = {"file": ("3d_flagged_graph.png", f.read(), "image/png")}
                payload = {"content": "Attached: 3D flagged-sample graph (PNG)."}
                response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
                response.raise_for_status()
        except (FileNotFoundError, IOError):
            pass
    elif graph_image_path and isinstance(graph_image_path, BytesIO):
        graph_image_path.seek(0)
        files = {"file": ("3d_flagged_graph.png", graph_image_path.getvalue(), "image/png")}
        payload = {"content": "Attached: 3D flagged-sample graph (PNG)."}
        response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
        response.raise_for_status()

    # Send multiple 3D graph images (one per flagged ID)
    if graph_images:
        for filename, graph_buffer in graph_images:
            if graph_buffer is None:
                continue
            graph_buffer.seek(0)
            out_name = filename if str(filename).strip() else "3d_flagged_graph.png"
            files = {"file": (out_name, graph_buffer.getvalue(), "image/png")}
            payload = {"content": f"Attached: 3D flagged-sample graph (PNG) for {out_name}."}
            response = requests.post(webhook_url, data=payload, files=files, timeout=timeout)
            response.raise_for_status()


def build_iem_screening_png_report(
    total_samples: int,
    normal_count: int,
    flagged_count: int,
    additional_signal_count: int,
    flagged_ids: list,
    additional_signal_ids: list,
    top1_predictions_df: Optional[pd.DataFrame] = None,
) -> BytesIO:
    """Generate a PNG report image with summary and top-1 prediction table."""
    has_table = top1_predictions_df is not None and not top1_predictions_df.empty
    row_count = int(min(len(top1_predictions_df), 15)) if has_table else 0
    fig_h = 5.8 + row_count * 0.36
    fig = plt.figure(figsize=(16, fig_h), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flagged_text = ", ".join(map(str, flagged_ids)) if flagged_ids else "None"
    additional_text = ", ".join(map(str, additional_signal_ids)) if additional_signal_ids else "None"

    ax.text(0.5, 0.97, "IEM Screening Report", ha="center", va="top", fontsize=20, fontweight="bold", color="#1e8f4e")
    ax.text(
        0.02,
        0.90,
        (
            f"Total: {total_samples} | Normal: {normal_count} | "
            f"Flagged: {flagged_count} | Additional Signals: {additional_signal_count}"
        ),
        fontsize=12,
        color="#1b1f23",
    )
    ax.text(0.02, 0.84, f"Model-flagged IDs: {flagged_text}", fontsize=11, color="#b00020")
    ax.text(0.02, 0.79, f"Additional signal IDs: {additional_text}", fontsize=11, color="#8a5a00")

    if has_table:
        table_df = top1_predictions_df.copy().head(15)
        preferred_cols = [
            "sample_id",
            "top_1_disease",
            "top_1_prob_%",
            "Marker",
            "Value",
            "MoM",
            "Lower_Cutoff",
            "Upper_Cutoff",
        ]
        table_cols = [c for c in preferred_cols if c in table_df.columns]
        if not table_cols:
            table_cols = list(table_df.columns)
        table_df = table_df[table_cols]

        # Ensure MoM is always shown with 2 decimal places in PNG report.
        if "MoM" in table_df.columns:
            table_df["MoM"] = pd.to_numeric(table_df["MoM"], errors="coerce").map(
                lambda v: "-" if pd.isna(v) else f"{float(v):.2f}"
            )

        table = ax.table(
            cellText=table_df.fillna("-").astype(str).values,
            colLabels=table_df.columns.tolist(),
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.07, 0.96, 0.66],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1e8f4e")
                cell.set_text_props(color="white", weight="bold")
            elif r == 1:
                cell.set_facecolor("#fdf1f4")
            elif r % 2 == 0:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")

    ax.text(0.98, 0.015, f"Generated: {generated_at}", ha="right", va="bottom", fontsize=10, color="#666666")

    png_buffer = BytesIO()
    fig.savefig(
        png_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)
    png_buffer.seek(0)
    return png_buffer


def build_additional_signal_ids_png(
    additional_signal_ids: list,
    additional_signal_details_df: Optional[pd.DataFrame] = None,
) -> BytesIO:
    """Generate a formal PNG report for additional pattern-signal sample IDs and details."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ids = [str(x) for x in additional_signal_ids] if additional_signal_ids else []
    row_count = len(ids)
    has_details = additional_signal_details_df is not None and not additional_signal_details_df.empty
    details_count = int(min(len(additional_signal_details_df), 20)) if has_details else 0
    fig_h = max(4.6, 3.4 + max(min(row_count, 30), details_count) * 0.30)
    fig = plt.figure(figsize=(14, fig_h), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Additional Pattern-Signal Sample IDs",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#8a5a00",
    )

    ax.text(
        0.02,
        0.89,
        f"Total additional pattern-signal IDs: {row_count}",
        fontsize=10.5,
        color="#1b1f23",
    )

    if has_details:
        table_df = additional_signal_details_df.copy().head(20)
        preferred_cols = [
            "Sample ID",
            "Pattern",
            "Support_%",
            "Marker",
            "Value",
            "MoM",
            "Lower_Cutoff",
            "Upper_Cutoff",
        ]
        table_cols = [c for c in preferred_cols if c in table_df.columns]
        if not table_cols:
            table_cols = list(table_df.columns)
        table_df = table_df[table_cols]

        for col in ["Support_%", "Value", "MoM", "Lower_Cutoff", "Upper_Cutoff"]:
            if col in table_df.columns:
                table_df[col] = pd.to_numeric(table_df[col], errors="coerce")
        if "Support_%" in table_df.columns:
            table_df["Support_%"] = table_df["Support_%"].map(
                lambda v: "-" if pd.isna(v) else f"{float(v):.2f}"
            )
        for col in ["Value", "Lower_Cutoff", "Upper_Cutoff"]:
            if col in table_df.columns:
                table_df[col] = table_df[col].map(
                    lambda v: "-" if pd.isna(v) else f"{float(v):.4f}"
                )
        if "MoM" in table_df.columns:
            table_df["MoM"] = table_df["MoM"].map(
                lambda v: "-" if pd.isna(v) else f"{float(v):.2f}"
            )

        # Wrap long disease/pattern names to avoid overlapping into neighboring cells.
        if "Pattern" in table_df.columns:
            table_df["Pattern"] = table_df["Pattern"].fillna("-").astype(str).map(
                lambda s: "\n".join(textwrap.wrap(s, width=28)) if s and s != "-" else "-"
            )

        # If wrapped text increases row height needs, slightly expand figure to keep readability.
        if "Pattern" in table_df.columns:
            max_lines = table_df["Pattern"].fillna("-").astype(str).map(lambda s: max(1, s.count("\n") + 1)).max()
            if pd.notna(max_lines) and int(max_lines) > 1:
                fig.set_size_inches(14, fig_h + 0.25 * (int(max_lines) - 1), forward=True)
    elif row_count > 0:
        table_df = pd.DataFrame({"No.": list(range(1, row_count + 1)), "Sample ID": ids})
    else:
        table_df = pd.DataFrame()

    if not table_df.empty:
        col_widths = None
        if has_details:
            # Keep Pattern wider and numeric columns compact for stable layout.
            col_widths = [0.125, 0.255, 0.095, 0.125, 0.10, 0.075, 0.1125, 0.1125]

        table = ax.table(
            cellText=table_df.fillna("-").astype(str).values,
            colLabels=[str(c) for c in table_df.columns.tolist()],
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.10, 0.96, 0.74],
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Increase row height for readability and prevent wrapped disease names from clipping.
        if has_details:
            pattern_col_idx = table_df.columns.tolist().index("Pattern") if "Pattern" in table_df.columns else None
            header_h = 0.085
            base_row_h = 0.090
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_height(header_h)
                    continue
                if pattern_col_idx is not None:
                    pattern_text = str(table_df.iloc[r - 1, pattern_col_idx])
                    line_count = max(1, pattern_text.count("\n") + 1)
                else:
                    line_count = 1
                cell.set_height(base_row_h * line_count)
        else:
            # For simple ID-only table, keep slightly taller rows.
            table.scale(1.0, 1.25)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#8a5a00")
                cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#fff6e8")
            else:
                cell.set_facecolor("#ffffff")
    else:
        ax.text(
            0.5,
            0.53,
            "No additional pattern-signal sample IDs were identified.",
            ha="center",
            va="center",
            fontsize=12,
            color="#1b1f23",
        )

    ax.text(0.98, 0.03, f"Generated: {generated_at}", ha="right", va="bottom", fontsize=9, color="#666666")

    png_buffer = BytesIO()
    fig.savefig(
        png_buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)
    png_buffer.seek(0)
    return png_buffer
