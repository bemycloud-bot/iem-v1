from __future__ import annotations

from io import BytesIO
from typing import Optional
from datetime import datetime
import textwrap
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import matplotlib.pyplot as plt


THAILAND_TZ = ZoneInfo("Asia/Bangkok")


def _now_th() -> datetime:
    return datetime.now(THAILAND_TZ)


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

    # Send initial message with embed only (no duplicate plain text)
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
        "timestamp": _now_th().isoformat(),
    }
    
    if additional_signal_ids:
        embed["fields"].append({
            "name": "Additional Pattern-Signal Sample IDs",
            "value": ", ".join(map(str, additional_signal_ids))[:1024],
            "inline": False,
        })
    
    payload = {
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

    # Build and send PNG for additional pattern-signal sample IDs (may be multiple pages)
    additional_pages = build_additional_signal_ids_png(
        additional_signal_ids=additional_signal_ids,
        additional_signal_details_df=additional_signal_details_df,
    )
    total_pages = len(additional_pages)
    for page_num, page_buf in enumerate(additional_pages, 1):
        page_buf.seek(0)
        suffix = f"_part{page_num}of{total_pages}" if total_pages > 1 else ""
        fname = f"additional_pattern_signal_ids{suffix}.png"
        part_label = f" (Part {page_num}/{total_pages})" if total_pages > 1 else ""
        content = f"Attached: Additional pattern-signal sample IDs{part_label} (PNG)."
        files = {"file": (fname, page_buf.getvalue(), "image/png")}
        response = requests.post(webhook_url, data={"content": content}, files=files, timeout=timeout)
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

    generated_at = _now_th().strftime("%Y-%m-%d %H:%M:%S")
    flagged_text = ", ".join(map(str, flagged_ids)) if flagged_ids else "None"
    additional_text = ", ".join(map(str, additional_signal_ids)) if additional_signal_ids else "None"
    flagged_text = textwrap.fill(flagged_text, width=120)
    additional_text = textwrap.fill(additional_text, width=120)

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

        # Wrap long disease text and limit to two lines to keep rows readable.
        if "top_1_disease" in table_df.columns:
            def _wrap_disease(value: object) -> str:
                text = "-" if pd.isna(value) else str(value)
                if text == "-":
                    return text
                lines = textwrap.wrap(text, width=24)
                if len(lines) > 2:
                    lines = lines[:2]
                    lines[-1] = lines[-1][:21].rstrip() + "..."
                return "\n".join(lines)

            table_df["top_1_disease"] = table_df["top_1_disease"].map(_wrap_disease)

        col_widths = None
        expected_cols = {
            "sample_id",
            "top_1_disease",
            "top_1_prob_%",
            "Marker",
            "Value",
            "MoM",
            "Lower_Cutoff",
            "Upper_Cutoff",
        }
        if expected_cols.issubset(set(table_df.columns)):
            col_widths = [0.125, 0.205, 0.125, 0.125, 0.105, 0.08, 0.1175, 0.1175]

        table = ax.table(
            cellText=table_df.fillna("-").astype(str).values,
            colLabels=table_df.columns.tolist(),
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.07, 0.96, 0.66],
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        disease_col_idx = table_df.columns.tolist().index("top_1_disease") if "top_1_disease" in table_df.columns else None
        if disease_col_idx is not None:
            line_counts = table_df.iloc[:, disease_col_idx].astype(str).map(lambda s: max(1, s.count("\n") + 1))
            extra_lines_total = int((line_counts - 1).clip(lower=0).sum())
            if extra_lines_total > 0:
                fig.set_size_inches(16, fig_h + 0.15 * extra_lines_total, forward=True)

            header_h = 0.082
            base_row_h = 0.082
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_height(header_h)
                else:
                    lc = int(line_counts.iloc[r - 1]) if (r - 1) < len(line_counts) else 1
                    cell.set_height(base_row_h * lc)
                cell.get_text().set_va("center")
                cell.get_text().set_ha("center")
        else:
            table.scale(1.0, 1.15)

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


_ADDITIONAL_ROWS_PER_PAGE = 7


def _render_additional_page(
    table_df: pd.DataFrame,
    total_ids: int,
    page_num: int,
    total_pages: int,
    generated_at: str,
    has_details: bool,
) -> BytesIO:
    """Render one paginated page of the additional pattern-signal report as PNG."""
    n_rows = len(table_df)
    fig_h = max(4.2, 3.0 + n_rows * 0.42)
    fig = plt.figure(figsize=(14, fig_h), dpi=150, facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    title = "Additional Pattern-Signal Sample IDs"
    if total_pages > 1:
        title += f"  (Part {page_num}/{total_pages})"
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=17, fontweight="bold", color="#8a5a00")
    ax.text(0.02, 0.89, f"Total additional pattern-signal IDs: {total_ids}", fontsize=10.5, color="#1b1f23")

    if not table_df.empty:
        col_widths = None
        if has_details:
            col_widths = [0.12, 0.29, 0.09, 0.12, 0.095, 0.07, 0.1075, 0.1075]

        # Expand figure height by total extra wrapped lines in Pattern column.
        if has_details and "Pattern" in table_df.columns:
            line_counts = table_df["Pattern"].fillna("-").astype(str).map(lambda s: max(1, s.count("\n") + 1))
            extra_lines_total = int((line_counts - 1).clip(lower=0).sum())
            if extra_lines_total > 0:
                fig.set_size_inches(14, fig_h + 0.14 * extra_lines_total, forward=True)
        else:
            line_counts = None

        table = ax.table(
            cellText=table_df.fillna("-").astype(str).values,
            colLabels=[str(c) for c in table_df.columns.tolist()],
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.10, 0.96, 0.76],
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        if has_details:
            pattern_col_idx = (
                table_df.columns.tolist().index("Pattern") if "Pattern" in table_df.columns else None
            )
            header_h = 0.090
            base_row_h = 0.090
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_height(header_h)
                else:
                    if pattern_col_idx is not None and line_counts is not None:
                        lc = int(line_counts.iloc[r - 1]) if (r - 1) < len(line_counts) else 1
                    else:
                        lc = 1
                    cell.set_height(base_row_h * lc)
                cell.get_text().set_va("center")
                cell.get_text().set_ha("center")
        else:
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
            0.5, 0.53,
            "No additional pattern-signal sample IDs were identified.",
            ha="center", va="center", fontsize=12, color="#1b1f23",
        )

    ax.text(0.98, 0.03, f"Generated: {generated_at}", ha="right", va="bottom", fontsize=9, color="#666666")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_additional_signal_ids_png(
    additional_signal_ids: list,
    additional_signal_details_df: Optional[pd.DataFrame] = None,
) -> list:
    """Return a list of PNG BytesIO pages. Splits into pages of 7 rows when there are more than 7 detail rows."""
    generated_at = _now_th().strftime("%Y-%m-%d %H:%M:%S")
    ids = [str(x) for x in additional_signal_ids] if additional_signal_ids else []
    total_ids = len(ids)
    has_details = additional_signal_details_df is not None and not additional_signal_details_df.empty

    if has_details:
        table_df = additional_signal_details_df.copy()
        preferred_cols = ["Sample ID", "Pattern", "Support_%", "Marker", "Value", "MoM", "Lower_Cutoff", "Upper_Cutoff"]
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

        if "Pattern" in table_df.columns:
            def _wrap_pattern(value: object) -> str:
                text = "-" if pd.isna(value) else str(value)
                if text == "-":
                    return text
                lines = textwrap.wrap(text, width=26)
                if len(lines) > 3:
                    lines = lines[:3]
                    lines[-1] = lines[-1][:23].rstrip() + "..."
                return "\n".join(lines)
            table_df["Pattern"] = table_df["Pattern"].fillna("-").astype(str).map(_wrap_pattern)

        chunks = [
            table_df.iloc[i : i + _ADDITIONAL_ROWS_PER_PAGE].reset_index(drop=True)
            for i in range(0, max(len(table_df), 1), _ADDITIONAL_ROWS_PER_PAGE)
        ]
    elif total_ids > 0:
        simple_df = pd.DataFrame({"No.": list(range(1, total_ids + 1)), "Sample ID": ids})
        chunks = [
            simple_df.iloc[i : i + _ADDITIONAL_ROWS_PER_PAGE].reset_index(drop=True)
            for i in range(0, max(len(simple_df), 1), _ADDITIONAL_ROWS_PER_PAGE)
        ]
        has_details = False
    else:
        return [_render_additional_page(pd.DataFrame(), 0, 1, 1, generated_at, has_details=False)]

    total_pages = len(chunks)
    return [
        _render_additional_page(chunk, total_ids, idx + 1, total_pages, generated_at, has_details=has_details)
        for idx, chunk in enumerate(chunks)
    ]
