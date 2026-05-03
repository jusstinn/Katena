"""BlackFiber commander dashboard — local Streamlit twin of the Foundry view.

Reads engagements.jsonl in real time, renders an ops-style display:
  - Big KPI counters at the top (total / neutralized / live)
  - Engagement timeline chart
  - Latest engagement detail card with full property dump
  - Engagement table (filtered by sidebar)
  - "Network status" indicator (mocked here; the real Foundry dashboard
     in Workshop will show actual link state)

Run:
    streamlit run dashboard.py

Auto-refreshes every 2s by default. Toggle in sidebar.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = PROJECT_ROOT / "engagements.jsonl"


@st.cache_data(ttl=1.5, show_spinner=False)
def load_engagements(path_str: str, _mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    latest: dict[str, dict] = {}
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                eng = rec["eng"]
                latest[eng["drone_id"]] = eng
            except (json.JSONDecodeError, KeyError):
                continue
    if not latest:
        return pd.DataFrame()
    df = pd.DataFrame(list(latest.values()))
    if "detection_timestamp" in df.columns:
        df["detection_dt"] = pd.to_datetime(df["detection_timestamp"], errors="coerce", utc=True)
        df = df.sort_values("detection_dt", ascending=False).reset_index(drop=True)
    return df


def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    neutralized = int(df.get("fiber_cut", pd.Series([], dtype=bool)).fillna(False).sum()) if not df.empty else 0
    avg_dur = (
        float(df["engagement_duration_s"].dropna().mean())
        if not df.empty and "engagement_duration_s" in df
        else 0.0
    )
    success_rate = (neutralized / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engagements", f"{total}", help="Total engagements logged")
    c2.metric("Neutralized", f"{neutralized}", f"{success_rate:.0f}% success")
    c3.metric("Avg duration", f"{avg_dur:.1f}s" if avg_dur > 0 else "—")
    c4.metric("Last engagement", _last_engagement_label(df))


def _last_engagement_label(df: pd.DataFrame) -> str:
    if df.empty or "detection_dt" not in df:
        return "—"
    latest = df["detection_dt"].max()
    if pd.isna(latest):
        return "—"
    delta = datetime.now(timezone.utc) - latest.to_pydatetime()
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return f"{secs // 3600}h ago"


def render_timeline(df: pd.DataFrame) -> None:
    if df.empty or "detection_dt" not in df:
        st.info("No engagements logged yet. Run the tracker to populate.")
        return
    bins = (
        df.set_index("detection_dt")
        .resample("1min")
        .size()
        .rename("engagements")
    )
    st.bar_chart(bins, height=180)


def render_latest(df: pd.DataFrame) -> None:
    if df.empty:
        return
    latest = df.iloc[0].to_dict()
    drone_id = latest.get("drone_id", "?")
    fiber_cut = latest.get("fiber_cut", False)
    signal_lost = latest.get("signal_lost", False)
    threat = latest.get("threat_level", "?")
    duration = latest.get("engagement_duration_s")

    status_color = "#1ea05a" if fiber_cut else "#c23a3a" if signal_lost else "#cba640"
    status_text = "NEUTRALIZED" if fiber_cut else "ACTIVE" if not signal_lost else "DEGRADING"

    st.markdown(
        f"""
        <div style="background:#15191f;padding:18px 22px;border-radius:10px;border-left:6px solid {status_color};">
            <div style="font-size:13px;color:#a0a8b4;letter-spacing:1px;">LATEST ENGAGEMENT</div>
            <div style="font-size:28px;font-weight:600;color:white;margin-top:4px;">{drone_id}
                <span style="font-size:14px;color:{status_color};margin-left:14px;letter-spacing:1.2px;">{status_text}</span>
            </div>
            <div style="color:#cdd2db;font-size:13px;margin-top:6px;">
                Threat: <b>{threat}</b>
                &nbsp;·&nbsp; Sensors: <b>{', '.join(latest.get('sensor_fusion', []))}</b>
                &nbsp;·&nbsp; Duration: <b>{(f'{duration:.2f}s' if isinstance(duration, (int, float)) else '—')}</b>
                &nbsp;·&nbsp; RF silence: <b>{'YES' if latest.get('rf_silence_confirmed') else 'NO'}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(df: pd.DataFrame, threat_filter: list[str], cut_filter: str) -> None:
    if df.empty:
        return
    view = df.copy()
    if threat_filter:
        view = view[view["threat_level"].isin(threat_filter)]
    if cut_filter == "Neutralized":
        view = view[view["fiber_cut"] == True]  # noqa: E712
    elif cut_filter == "Active / lost":
        view = view[view["fiber_cut"] != True]  # noqa: E712

    keep_cols = [
        "drone_id",
        "detection_timestamp",
        "threat_level",
        "fiber_cut",
        "signal_lost",
        "rf_silence_confirmed",
        "engagement_duration_s",
        "signal_strength",
        "pan_angle",
        "tilt_angle",
        "notes",
    ]
    cols = [c for c in keep_cols if c in view.columns]
    st.dataframe(view[cols], hide_index=True, use_container_width=True, height=320)


def main() -> None:
    st.set_page_config(
        page_title="BlackFiber · FOG Drone Neutralizer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.6rem; }
            [data-testid="stMetricValue"] { font-size: 2rem; }
            [data-testid="stMetricDelta"] { color: #1ea05a; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### BlackFiber")
        st.caption("FOG Drone Fiber-Tether Neutralization")
        log_path_str = st.text_input("Engagements log", str(DEFAULT_LOG))
        auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)
        threat_filter = st.multiselect(
            "Threat level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
        cut_filter = st.radio(
            "Status",
            ["All", "Neutralized", "Active / lost"],
            horizontal=False,
        )
        st.divider()
        st.caption(
            "This local view mirrors the Foundry Workshop dashboard. "
            "When the Jetson syncs to Foundry, the same `DroneEngagement` "
            "objects show up there as the common operating picture."
        )

    log_path = Path(log_path_str)
    mtime = log_path.stat().st_mtime if log_path.exists() else 0.0
    df = load_engagements(str(log_path), mtime)

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown("## Counter-FOG Operations")
        st.caption(f"Source: `{log_path}` · last update {datetime.now().strftime('%H:%M:%S')}")
    with header_right:
        live = "LIVE" if auto_refresh else "PAUSED"
        color = "#1ea05a" if auto_refresh else "#c23a3a"
        st.markdown(
            f"<div style='text-align:right;'><span style='color:{color};font-weight:600;letter-spacing:2px;'>● {live}</span></div>",
            unsafe_allow_html=True,
        )

    render_kpis(df)
    st.divider()
    render_latest(df)
    st.divider()

    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown("#### Timeline")
        render_timeline(df)
    with cols[1]:
        st.markdown("#### Engagements")
        render_table(df, threat_filter, cut_filter)

    if auto_refresh:
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
