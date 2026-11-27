"""Streamlit dashboard for EgoRoad inference telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import streamlit as st


TELEMETRY_FILE = "situational_awareness.jsonl"
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def discover_runs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path.parent for path in root.rglob(TELEMETRY_FILE) if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def parse_records(text: str, source_name: str) -> List[Dict]:
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source_name}:{line_number}: invalid JSON ({error.msg})"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"{source_name}:{line_number}: expected a JSON object")
        if not isinstance(record.get("features"), dict) or not isinstance(
            record.get("decision"), dict
        ):
            raise ValueError(
                f"{source_name}:{line_number}: expected 'features' and 'decision' objects"
            )
        records.append(record)
    return records


def load_records(path: Path) -> List[Dict]:
    return parse_records(path.read_text(encoding="utf-8-sig"), str(path))


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def display_number(value, suffix="", precision=1):
    number = finite_number(value)
    return f"{number:.{precision}f}{suffix}" if number is not None else "—"


def display_percent(value):
    number = finite_number(value)
    return f"{number * 100:.0f}%" if number is not None else "—"


def find_media(run_dir: Path, source: str):
    if not source or source.isdigit():
        return None
    media_path = run_dir / Path(source).name
    if media_path.is_file() and media_path.suffix.lower() in (
        VIDEO_SUFFIXES | IMAGE_SUFFIXES
    ):
        return media_path
    return None


def style_axis(axis):
    axis.set_facecolor("#0e1117")
    axis.tick_params(colors="#94a3b8")
    axis.xaxis.label.set_color("#cbd5e1")
    axis.yaxis.label.set_color("#cbd5e1")
    for spine in axis.spines.values():
        spine.set_color("#334155")
    axis.grid(color="#334155", alpha=0.45)


def select_frame(frame_values: List[int]) -> int:
    minimum_frame, maximum_frame = min(frame_values), max(frame_values)
    if minimum_frame == maximum_frame:
        st.sidebar.caption(f"Only frame {minimum_frame} is available.")
        return minimum_frame
    return st.sidebar.slider(
        "Frame",
        min_value=minimum_frame,
        max_value=maximum_frame,
        value=maximum_frame,
    )


def show_media(media_path: Optional[Path], record: Dict):
    if media_path is None:
        st.info("No matching annotated image or video was saved in this run.")
        return

    if media_path.suffix.lower() in IMAGE_SUFFIXES:
        st.image(str(media_path), use_container_width=True)
        return

    import cv2

    capture = cv2.VideoCapture(str(media_path))
    try:
        frame_number = record.get("frame", 1)
        try:
            frame_index = max(0, int(frame_number) - 1)
        except (TypeError, ValueError):
            frame_index = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if success:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Annotated video frame {frame_number}")
        else:
            st.warning(f"Could not read frame {frame_number} from {media_path.name}.")
    finally:
        capture.release()


def main():
    st.set_page_config(
        page_title="EgoRoad | Situational Awareness",
        page_icon="🚘",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .eyebrow {color:#38bdf8; font-size:.78rem; font-weight:700;
                  letter-spacing:.16em; text-transform:uppercase;}
        .hero {border:1px solid rgba(148,163,184,.22); border-radius:18px;
               padding:1.4rem 1.6rem; margin-bottom:1.2rem;
               background:linear-gradient(120deg,rgba(14,165,233,.12),
               rgba(15,23,42,.16));}
        .hero h1 {margin:.3rem 0 .35rem; font-size:2rem;}
        .hero p {color:#94a3b8; margin:0;}
        .status-pill {display:inline-block; padding:.3rem .7rem;
                      border-radius:999px; font-size:.78rem; font-weight:700;
                      letter-spacing:.04em; background:rgba(56,189,248,.14);
                      color:#7dd3fc;}
        </style>
        <div class="hero">
          <div class="eyebrow">EgoRoad perception suite</div>
          <h1>Situational Awareness Dashboard</h1>
          <p>Review YOLOPv2 detections, road features, and fuzzy safety advisories.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Run selection")
        root_text = st.text_input("Results folder", value="runs/detect")
        uploaded_telemetry = st.file_uploader(
            "Or upload telemetry (.jsonl)",
            type=["jsonl"],
        )
        uploaded_frame = st.file_uploader(
            "Annotated frame image (optional)",
            type=["bmp", "jpeg", "jpg", "png", "tif", "tiff"],
        )
        if st.button("Refresh runs", use_container_width=True):
            st.rerun()

    run_dir = None
    if uploaded_telemetry is not None:
        try:
            records = parse_records(
                uploaded_telemetry.getvalue().decode("utf-8-sig"),
                uploaded_telemetry.name,
            )
        except (UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not load uploaded telemetry: {error}")
            st.stop()
    else:
        root = Path(root_text).expanduser()
        if not root.is_dir():
            st.error(f"Results folder does not exist: {root}")
            st.stop()

        runs = discover_runs(root)
        if not runs:
            st.info(
                f"No {TELEMETRY_FILE} files found under {root}. "
                "Upload a telemetry JSONL file in the sidebar, or run inference "
                "with telemetry enabled."
            )
            st.stop()

        run_dir = st.sidebar.selectbox(
            "Inference run",
            runs,
            format_func=lambda path: f"{path.name}  ·  {path.parent}",
        )
        telemetry_path = run_dir / TELEMETRY_FILE
        try:
            records = load_records(telemetry_path)
        except (OSError, ValueError) as error:
            st.error(f"Could not load this run: {error}")
            st.stop()

    if not records:
        st.info("This run has no frame records yet.")
        st.stop()

    sources = sorted({str(record.get("source", "unknown")) for record in records})
    source = st.sidebar.selectbox(
        "Input source",
        sources,
        format_func=lambda value: f"{Path(value).name or value}  ·  {value}",
    )
    source_records = [
        record for record in records if str(record.get("source", "unknown")) == source
    ]
    source_records.sort(key=lambda record: int(record.get("frame", 0)))

    frame_values = [int(record.get("frame", 0)) for record in source_records]
    selected_frame = select_frame(frame_values)
    record = min(
        source_records,
        key=lambda item: abs(int(item.get("frame", 0)) - selected_frame),
    )
    features = record["features"]
    decision = record["decision"]
    frame = int(record.get("frame", 0))

    scores = [
        finite_number(item["decision"].get("sas"))
        for item in source_records
    ]
    valid_scores = [score for score in scores if score is not None]
    valid_frames = sum(
        bool(item["features"].get("feature_valid")) for item in source_records
    )
    source_label = "source" if len(sources) == 1 else "sources"
    awareness_counts = {}
    for item in source_records:
        level = str(item["decision"].get("awareness", "UNKNOWN")).upper()
        awareness_counts[level] = awareness_counts.get(level, 0) + 1

    st.markdown(
        f'<span class="status-pill">{len(source_records):,} frames</span>'
        f'&nbsp;&nbsp; <span class="status-pill">{valid_frames:,} valid feature frames</span>'
        f'&nbsp;&nbsp; <span class="status-pill">{len(sources):,} input {source_label}</span>',
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric(
        "Situational awareness",
        f'{display_number(decision.get("sas"))} / 10',
        str(decision.get("awareness", "Unknown")).title(),
    )
    metrics[1].metric(
        "Braking advisory",
        display_percent(decision.get("braking_intensity")),
        str(decision.get("steering", "—")).title(),
    )
    metrics[2].metric(
        "Lane deviation",
        display_number(features.get("lane_deviation_m"), " m", 2),
        str(features.get("lane_state", "unknown")).title(),
    )
    metrics[3].metric(
        "Nearest object / TTC",
        display_number(features.get("ttc_s"), " s"),
        str(features.get("nearest_object_class", "none")).title(),
    )

    left, right = st.columns([1.55, 1])
    with left:
        st.subheader(f"Annotated output · frame {frame}")
        if uploaded_frame is not None:
            st.image(uploaded_frame.getvalue(), use_container_width=True)
            st.caption(f"Uploaded frame: {uploaded_frame.name}")
        else:
            show_media(
                find_media(run_dir, source) if run_dir is not None else None,
                record,
            )
    with right:
        st.subheader("Frame assessment")
        if not features.get("feature_valid", False):
            st.warning("LIMITED DATA — lane and drivable-area features are unavailable.")
        st.metric("Drivable corridor", display_percent(features.get("drivable_ratio")))
        st.progress(
            min(1.0, max(0.0, finite_number(features.get("drivable_ratio")) or 0.0)),
            text="Drivable-area coverage",
        )
        st.metric("Pothole severity", display_number(features.get("pothole_severity"), "/ 1", 2))
        st.progress(
            min(1.0, max(0.0, finite_number(features.get("pothole_severity")) or 0.0))
        )
        st.metric("Surface hazard", display_number(decision.get("surface_hazard"), " / 10"))
        st.metric("Clearance risk", display_number(decision.get("clearance_risk"), " / 10"))

    st.subheader("Run timeline")
    figure, axis = plt.subplots(figsize=(12, 3.2))
    figure.patch.set_facecolor("#0e1117")
    axis.plot(
        frame_values,
        [finite_number(item["decision"].get("sas")) for item in source_records],
        label="Smoothed SAS",
        color="#38bdf8",
        linewidth=2,
    )
    axis.plot(
        frame_values,
        [finite_number(item["decision"].get("sas_raw")) for item in source_records],
        label="Raw SAS",
        color="#a78bfa",
        linewidth=1,
        alpha=0.75,
    )
    axis.set(xlabel="Frame", ylabel="Situational awareness", ylim=(1, 10))
    axis.legend(facecolor="#0e1117", labelcolor="#cbd5e1", framealpha=0.8)
    style_axis(axis)
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    feature_columns = st.columns(2)
    with feature_columns[0]:
        figure, axis = plt.subplots(figsize=(6, 2.7))
        figure.patch.set_facecolor("#0e1117")
        axis.plot(
            frame_values,
            [
                finite_number(item["features"].get("lane_deviation_m"))
                for item in source_records
            ],
            color="#38bdf8",
            linewidth=1.5,
        )
        axis.set(xlabel="Frame", ylabel="Lane deviation (m)")
        style_axis(axis)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with feature_columns[1]:
        figure, axis = plt.subplots(figsize=(6, 2.7))
        figure.patch.set_facecolor("#0e1117")
        axis.plot(
            frame_values,
            [
                finite_number(item["features"].get("drivable_ratio"))
                for item in source_records
            ],
            label="Drivable ratio",
            color="#34d399",
            linewidth=1.5,
        )
        axis.plot(
            frame_values,
            [
                finite_number(item["features"].get("pothole_severity"))
                for item in source_records
            ],
            label="Pothole severity",
            color="#fb7185",
            linewidth=1.5,
        )
        axis.set(xlabel="Frame", ylabel="Ratio / severity", ylim=(0, 1))
        axis.legend(facecolor="#0e1117", labelcolor="#cbd5e1", framealpha=0.8)
        style_axis(axis)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    detail, summary = st.columns([1.5, 1])
    with detail:
        with st.expander("Selected frame details"):
            st.json({"features": features, "decision": decision})
    with summary:
        st.markdown("**Awareness distribution**")
        st.dataframe(
            [
                {"Level": level.title(), "Frames": count}
                for level, count in sorted(awareness_counts.items())
            ],
            hide_index=True,
            use_container_width=True,
        )
        if valid_scores:
            st.caption(
                f"Average SAS: {sum(valid_scores) / len(valid_scores):.2f} / 10"
            )

    st.caption(
        "Distance, TTC, and fuzzy advisories are experimental estimates, "
        "not safety-certified driving controls."
    )


if __name__ == "__main__":
    main()
