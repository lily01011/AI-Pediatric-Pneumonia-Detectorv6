"""
=============================================================================
  AI Pediatric Pneumonia Detector — Treatment Plan Page
  File   : pages/Treatment.py
  Design : 55% White | 30% Blue (#1e90ff) | 15% Navy (#001f3f) | Text #000000
=============================================================================
"""

import streamlit as st
import pandas as pd
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Treatment Plan",
    page_icon=":material/medication:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (minimal – only what Streamlit cannot do natively) ────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"]          { display: none; }
[data-testid="collapsedControl"]   { display: none; }
#MainMenu, footer, header          { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; }

/* ── Section cards ── */
.tx-card {
    background: #f4f7ff;
    border: 1.5px solid #d0e0ff;
    border-left: 4px solid #1e90ff;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}
.tx-card-navy {
    background: #f0f2f8;
    border: 1.5px solid #bbc8e8;
    border-left: 4px solid #001f3f;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
}

/* ── Section headings ── */
.tx-section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #001f3f;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Check table ── */
.check-table-header {
    background: #001f3f;
    color: #ffffff;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    font-weight: 700;
    font-size: 0.92rem;
    margin-top: 1.2rem;
}
.check-table-wrap {
    border: 1.5px solid #001f3f;
    border-top: none;
    border-radius: 0 0 6px 6px;
    overflow: hidden;
    margin-bottom: 1.4rem;
}
.check-table-wrap table {
    width: 100%;
    border-collapse: collapse;
}
.check-table-wrap th {
    background: #e8eef8;
    color: #001f3f;
    padding: 0.45rem 0.7rem;
    font-size: 0.82rem;
    border-bottom: 1.5px solid #001f3f;
    border-right: 1px solid #c0cfe8;
    text-align: left;
}
.check-table-wrap td {
    padding: 0.4rem 0.7rem;
    font-size: 0.82rem;
    border-bottom: 1px solid #d0ddf0;
    border-right: 1px solid #e0eaff;
    color: #000000;
    background: #ffffff;
}
.check-table-wrap tr:last-child td { border-bottom: none; }

/* ── Divider ── */
.tx-divider {
    height: 2px;
    background: linear-gradient(90deg, #1e90ff 0%, #001f3f 60%, transparent 100%);
    margin: 1.4rem 0;
    border-radius: 2px;
}

/* ── Timer badge ── */
.timer-badge {
    display: inline-block;
    background: #001f3f;
    color: #fff;
    border-radius: 20px;
    padding: 0.22rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state bootstrap ──────────────────────────────────────────────────
for key, default in {
    "hospital_meds":        [],
    "home_meds":            [],
    "warning_signs":        [],
    "extra_columns":        [],       # doctor-added column names
    "extra_col_inputs":     [],       # pending input fields for new columns
    "timer_active":         False,
    "timer_start":          None,
    "last_check_time":      None,     # timestamp of last table generation
    "timer_duration":       0,        # interval in seconds
    "current_table":        0,
    "check_tables":         [],       # list of dicts {index, rows}
    "show_timer_sel":       False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# HELPER FUNCTIONS  — must be defined BEFORE any usage
# =============================================================================

def _default_check_rows():
    """
    Returns the default row structure for a new check table.
    Each row represents one vital/observation field the nurse fills in.
    Extra columns are appended dynamically from st.session_state.extra_columns.
    """
    base_rows = [
        {"field": "Temperature (°C)",      "value": "", "note": "", "extra": {}},
        {"field": "Heart Rate (bpm)",       "value": "", "note": "", "extra": {}},
        {"field": "Respiratory Rate (/min)","value": "", "note": "", "extra": {}},
        {"field": "SpO₂ (%)",               "value": "", "note": "", "extra": {}},
        {"field": "Blood Pressure (mmHg)",  "value": "", "note": "", "extra": {}},
        {"field": "Consciousness Level",    "value": "", "note": "", "extra": {}},
        {"field": "Medication Given",       "value": "", "note": "", "extra": {}},
    ]
    # Pre-populate any existing extra columns
    for row in base_rows:
        for col in st.session_state.extra_columns:
            row["extra"][col] = ""
    return base_rows


def _render_check_tables():
    """
    Renders ALL generated check tables with editable nurse input fields.
    This function is always called independently of monitoring state —
    tables already created remain visible even after monitoring stops.
    """
    for tbl in st.session_state.check_tables:
        t_idx = tbl["index"]
        rows  = tbl["rows"]

        st.markdown(
            f"<div class='check-table-header'>:material/assignment: Check Table {t_idx}</div>",
            unsafe_allow_html=True,
        )

        # Build column headers
        base_headers = ["Observation", "Value", "Notes"]
        extra_headers = st.session_state.extra_columns

        # Header row
        header_cols = st.columns([3, 2, 3] + [2] * len(extra_headers))
        for col, label in zip(header_cols, base_headers + extra_headers):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;"
                f"padding:0.3rem 0;'>{label}</div>",
                unsafe_allow_html=True,
            )

        # Data rows
        for r_idx, row in enumerate(rows):
            data_cols = st.columns([3, 2, 3] + [2] * len(extra_headers))

            # Field label (read-only)
            data_cols[0].markdown(
                f"<div style='font-size:0.82rem; padding:0.45rem 0; color:#001f3f;'>"
                f"{row['field']}</div>",
                unsafe_allow_html=True,
            )

            # Value input
            row["value"] = data_cols[1].text_input(
                "v",
                value=row.get("value", ""),
                label_visibility="collapsed",
                key=f"ct{t_idx}_r{r_idx}_val",
            )

            # Notes input
            row["note"] = data_cols[2].text_input(
                "n",
                value=row.get("note", ""),
                label_visibility="collapsed",
                key=f"ct{t_idx}_r{r_idx}_note",
            )

            # Extra column inputs
            for c_idx, col_name in enumerate(extra_headers):
                row["extra"] = row.get("extra", {})
                row["extra"][col_name] = data_cols[3 + c_idx].text_input(
                    col_name,
                    value=row["extra"].get(col_name, ""),
                    label_visibility="collapsed",
                    key=f"ct{t_idx}_r{r_idx}_ex{c_idx}",
                )

        st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE HEADER
# =============================================================================
col_icon, col_title = st.columns([0.07, 0.93])
with col_icon:
    st.markdown(
        "<span style='font-size:2.2rem; color:#001f3f;'>:material/medication:</span>",
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        "<h1 style='color:#001f3f; font-size:1.8rem; font-weight:800; margin:0;'>Treatment Plan</h1>"
        "<p style='color:#1e90ff; margin:0; font-size:0.95rem;'>Select a treatment mode and complete the plan below.</p>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)

# Patient selector
patient = st.selectbox(
    ":material/person: Select Patient",
    ["— Select Patient —", "Patient 1", "Patient 2", "Patient 3"],
)

st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# HOSPITALIZATION STATUS — radio
# =============================================================================
st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
st.markdown(
    "<div class='tx-section-title'>:material/local_hospital: &nbsp;Hospitalization Status</div>",
    unsafe_allow_html=True,
)
plan = st.radio(
    "plan",
    ["— Select a plan —", "Patient Hospitalized", "Home Treatment"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Empty-state placeholder ──────────────────────────────────────────────────
if plan == "— Select a plan —":
    st.info(":material/info: Please select a treatment plan above to begin.", icon=None)
    st.stop()


# =============================================================================
# ===================== HOSPITALIZED TREATMENT SECTION ========================
# =============================================================================
if plan == "Patient Hospitalized":

    # ── A. In-Hospital Instructions ──────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/assignment: &nbsp;In-Hospital Treatment Plan</div>",
        unsafe_allow_html=True,
    )

    st.text_area(
        ":material/edit_note: Treatment Instructions for Nurses",
        placeholder="Enter detailed nursing care instructions, surveillance schedule, special precautions…",
        height=110,
        key="hosp_instructions",
    )

    st.text_area(
        ":material/notes: Medical Progress Notes",
        placeholder="Add initial progress observations…",
        height=90,
        key="hosp_progress",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── B. Prescribed Medications ─────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/medication: &nbsp;Prescribed Medications</div>",
        unsafe_allow_html=True,
    )

    if st.button(":material/add: Add Medication", key="add_hosp_med"):
        st.session_state.hospital_meds.append(
            {"name": "", "dosage": "", "schedule": "", "duration": ""}
        )

    if st.session_state.hospital_meds:
        hdr = st.columns([3, 2, 3, 2, 1])
        for col, label in zip(hdr, ["Medicine Name", "Dosage (g/day)", "Time Schedule", "Duration (days)", ""]):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{label}</div>",
                unsafe_allow_html=True,
            )

        to_del = None
        for i, med in enumerate(st.session_state.hospital_meds):
            c1, c2, c3, c4, c5 = st.columns([3, 2, 3, 2, 1])
            med["name"]     = c1.text_input("n", value=med["name"],     placeholder="e.g. Amoxicillin",   label_visibility="collapsed", key=f"hm_n_{i}")
            med["dosage"]   = c2.text_input("d", value=med["dosage"],   placeholder="e.g. 1.5",           label_visibility="collapsed", key=f"hm_d_{i}")
            med["schedule"] = c3.text_input("s", value=med["schedule"], placeholder="e.g. 08:00 / 20:00", label_visibility="collapsed", key=f"hm_s_{i}")
            med["duration"] = c4.text_input("u", value=med["duration"], placeholder="e.g. 7",             label_visibility="collapsed", key=f"hm_u_{i}")
            if c5.button(":material/delete:", key=f"hm_del_{i}"):
                to_del = i
        if to_del is not None:
            st.session_state.hospital_meds.pop(to_del)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── C. Nurse Check Interval & Dynamic Check Tables ────────────────────────
    # =========================================================================
    #   NON-BLOCKING MONITORING ENGINE
    #   - Uses time.time() to track elapsed time; no while loops or sleep().
    #   - On every rerun, computes how many intervals have passed and appends
    #     new check tables accordingly.
    #   - st.rerun() is called ONLY when monitoring is active, to keep the
    #     countdown live without blocking the UI.
    # =========================================================================

    st.markdown("<div class='tx-card-navy'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/schedule: &nbsp;Nurse Monitoring Interval &amp; Check Tables</div>",
        unsafe_allow_html=True,
    )

    col_num, col_unit, col_start = st.columns([2, 2, 2])

    with col_num:
        interval_val = st.number_input(
            "Check every (enter a number)",
            min_value=0,
            value=0,
            step=1,
            key="interval_val",
        )

    interval_seconds = 0
    interval_unit    = None

    if interval_val and interval_val > 0:
        with col_unit:
            interval_unit = st.selectbox(
                "Unit",
                ["seconds", "minutes", "hours"],
                key="interval_unit",
            )
        multipliers      = {"seconds": 1, "minutes": 60, "hours": 3600}
        interval_seconds = int(interval_val) * multipliers[interval_unit]

        with col_start:
            st.markdown("<br>", unsafe_allow_html=True)
            if not st.session_state.timer_active:
                if st.button(":material/play_arrow: Start Monitoring", key="start_timer"):
                    now = time.time()
                    st.session_state.timer_active    = True
                    st.session_state.timer_start     = now
                    st.session_state.last_check_time = now
                    st.session_state.timer_duration  = interval_seconds
                    st.session_state.current_table   = 1
                    # Generate the first check table immediately on start
                    st.session_state.check_tables.append(
                        {"index": 1, "rows": _default_check_rows()}
                    )
                    st.rerun()
            else:
                if st.button(":material/stop: Stop Monitoring", key="stop_timer"):
                    st.session_state.timer_active = False
                    # Do NOT clear check_tables — existing tables stay visible
                    st.rerun()

    # ── Non-blocking interval check & countdown ──────────────────────────────
    # LOGIC:
    #   1. Compute elapsed time since last table was generated.
    #   2. If interval elapsed → append new table to session state (NO rerun yet).
    #   3. Render countdown badge.
    #   4. Render ALL tables (including the one just added).
    #   5. THEN sleep(1) + rerun → keeps countdown live without blocking render.
    #
    # Key fix: do NOT call st.rerun() immediately after appending a table.
    # Doing so restarts the script before _render_check_tables() executes,
    # which is why tables only appeared after Stop was pressed.
    if st.session_state.timer_active and st.session_state.timer_duration > 0:
        now           = time.time()
        elapsed_since = now - st.session_state.last_check_time

        # Generate new table if interval elapsed — NO rerun here
        if elapsed_since >= st.session_state.timer_duration:
            new_idx = len(st.session_state.check_tables) + 1
            st.session_state.check_tables.append(
                {"index": new_idx, "rows": _default_check_rows()}
            )
            st.session_state.current_table   = new_idx
            st.session_state.last_check_time = now   # reset interval clock

        # Recalculate after possible table addition
        elapsed_since  = time.time() - st.session_state.last_check_time
        remaining      = int(st.session_state.timer_duration - elapsed_since)
        remaining      = max(remaining, 0)
        next_table_idx = len(st.session_state.check_tables) + 1

        # Display live countdown
        st.markdown(
            f"<p style='color:#001f3f; font-size:0.88rem; margin-top:0.6rem;'>"
            f":material/timer: &nbsp;Next check table "
            f"<strong>(Table {next_table_idx})</strong> in "
            f"<span class='timer-badge'>{remaining}s</span></p>",
            unsafe_allow_html=True,
        )

    # ── Dynamic custom column manager ────────────────────────────────────────
    # The doctor can add an unlimited number of custom monitoring columns.
    # Each click on "Add Column" appends a new persistent input field.
    # All columns are stored in st.session_state.extra_columns and are
    # reflected in every existing and future check table.
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='color:#001f3f; font-weight:700; font-size:0.9rem; margin:1rem 0 0.5rem;'>"
        ":material/add_chart: &nbsp;Custom Monitoring Columns (Doctor)</div>",
        unsafe_allow_html=True,
    )

    # Button to add a new blank input slot
    if st.button(":material/add: Add Column Field", key="add_col_field_btn"):
        st.session_state.extra_col_inputs.append("")
        st.rerun()

    # Render each pending input slot
    to_confirm = []
    for i, pending in enumerate(st.session_state.extra_col_inputs):
        c_input, c_confirm = st.columns([5, 1])
        col_val = c_input.text_input(
            f"Column name {i + 1}",
            value=pending,
            placeholder="e.g. Blood Pressure, Urine Output…",
            label_visibility="collapsed",
            key=f"col_input_{i}",
        )
        st.session_state.extra_col_inputs[i] = col_val

        if c_confirm.button(":material/check: Save", key=f"col_confirm_{i}"):
            to_confirm.append(i)

    # Process confirmed columns
    for idx in sorted(to_confirm, reverse=True):
        col_name = st.session_state.extra_col_inputs[idx].strip()
        if col_name and col_name not in st.session_state.extra_columns:
            st.session_state.extra_columns.append(col_name)
            # Append the new column to every row in all existing tables
            for tbl in st.session_state.check_tables:
                for row in tbl["rows"]:
                    row.setdefault("extra", {})
                    row["extra"][col_name] = ""
        # Remove the input slot that was confirmed
        st.session_state.extra_col_inputs.pop(idx)
        st.rerun()

    # Show confirmed columns list
    if st.session_state.extra_columns:
        st.markdown(
            "<div style='font-size:0.82rem; color:#1e90ff; margin:0.4rem 0;'>"
            "Active custom columns: " + " · ".join(
                f"<strong>{c}</strong>" for c in st.session_state.extra_columns
            ) + "</div>",
            unsafe_allow_html=True,
        )
        if st.button(":material/delete_sweep: Clear All Custom Columns", key="clear_cols"):
            st.session_state.extra_columns     = []
            st.session_state.extra_col_inputs  = []
            st.rerun()

    # ── Render all generated check tables (always, independent of timer state)
    # Tables render HERE first — then we schedule the next countdown tick.
    _render_check_tables()

    # ── Schedule countdown tick AFTER rendering ──────────────────────────────
    # sleep(1) + rerun here means: render everything above first, THEN restart.
    # This is the fix — rerun inside the timer block above was the bug that
    # caused tables to never display while monitoring was active.
    if st.session_state.timer_active and st.session_state.timer_duration > 0:
        time.sleep(1)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end monitoring card

    # ── D. Discharge Planning ─────────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/exit_to_app: &nbsp;Discharge Planning</div>",
        unsafe_allow_html=True,
    )

    st.text_area(
        ":material/fact_check: Discharge Conditions",
        placeholder="Clinical conditions required for safe discharge…",
        height=90,
        key="discharge_cond",
    )

    st.markdown(
        "<div style='color:#001f3f; font-weight:700; font-size:0.87rem; margin:0.8rem 0 0.4rem;'>"
        "Required Stability Indicators</div>",
        unsafe_allow_html=True,
    )
    st.checkbox("SpO\u2082 > 94%",           value=True, key="dc_spo2")
    st.checkbox("Fever-free for 24 h",        value=True, key="dc_fever")
    st.checkbox("Tolerating oral feeds",                  key="dc_feeds")
    st.checkbox("Improved chest X-ray",                   key="dc_xray_ok")

    st.file_uploader(
        ":material/upload: Upload updated chest X-ray",
        type=["png", "jpg", "jpeg", "dcm"],
        key="discharge_xray",
    )

    st.text_area(
        ":material/notes: Clinical Notes",
        placeholder="Additional observations about this case…",
        height=90,
        key="dc_notes",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
    if st.button(
        ":material/save: Save Treatment Plan",
        use_container_width=True,
        key="save_hosp",
        type="primary",
    ):
        st.success(":material/check_circle: In-hospital treatment plan saved successfully. (Placeholder)")

# =============================================================================
# ===================== HOME TREATMENT SECTION ================================
# =============================================================================
elif plan == "Home Treatment":

    # ── A. Prescribed Medications ─────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/medication: &nbsp;Prescribed Medications</div>",
        unsafe_allow_html=True,
    )

    if st.button(":material/add: Add Medication", key="add_home_med"):
        st.session_state.home_meds.append(
            {"name": "", "dosage": "", "schedule": "", "duration": ""}
        )

    if st.session_state.home_meds:
        hdr = st.columns([3, 2, 3, 2, 1])
        for col, label in zip(hdr, ["Medicine Name", "Dosage (g/day)", "Time Schedule", "Duration (days)", ""]):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{label}</div>",
                unsafe_allow_html=True,
            )
        to_del2 = None
        for i, med in enumerate(st.session_state.home_meds):
            c1, c2, c3, c4, c5 = st.columns([3, 2, 3, 2, 1])
            med["name"]     = c1.text_input("n", value=med["name"],     placeholder="e.g. Amoxicillin",  label_visibility="collapsed", key=f"hom_n_{i}")
            med["dosage"]   = c2.text_input("d", value=med["dosage"],   placeholder="e.g. 1.5",          label_visibility="collapsed", key=f"hom_d_{i}")
            med["schedule"] = c3.text_input("s", value=med["schedule"], placeholder="e.g. 08:00 / 20:00",label_visibility="collapsed", key=f"hom_s_{i}")
            med["duration"] = c4.text_input("u", value=med["duration"], placeholder="e.g. 7",            label_visibility="collapsed", key=f"hom_u_{i}")
            if c5.button(":material/delete:", key=f"hom_del_{i}"):
                to_del2 = i
        if to_del2 is not None:
            st.session_state.home_meds.pop(to_del2)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── B. Next Appointment ───────────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/calendar_today: &nbsp;Next Appointment</div>",
        unsafe_allow_html=True,
    )
    st.date_input(
        ":material/event: Next appointment date",
        value=None,
        format="DD/MM/YYYY",
        key="home_appt_date",
    )
    st.text_area(
        ":material/edit_note: Follow-up Clinical Notes",
        placeholder="Indicators to check during the next visit. Instructions for the patient's family.",
        height=100,
        key="home_followup",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── C. Emergency Warning Signs ────────────────────────────────────────────
    st.markdown("<div class='tx-card-navy'>", unsafe_allow_html=True)
    col_w, col_wb = st.columns([3, 2])
    with col_w:
        st.markdown(
            "<div class='tx-section-title'>:material/warning: &nbsp;Emergency Warning Signs</div>",
            unsafe_allow_html=True,
        )
    with col_wb:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(":material/add: Add Warning Sign", key="add_warn"):
            st.session_state.warning_signs.append({"mark": "", "instruction": ""})

    if st.session_state.warning_signs:
        wh = st.columns([3, 4, 1])
        for col, lbl in zip(wh, ["Danger Sign", "Parent Instruction", ""]):
            col.markdown(
                f"<div style='color:#001f3f; font-size:0.82rem; font-weight:700;'>{lbl}</div>",
                unsafe_allow_html=True,
            )
    to_del_w = None
    for i, sign in enumerate(st.session_state.warning_signs):
        c1, c2, c3 = st.columns([3, 4, 1])
        sign["mark"]        = c1.text_input("m", value=sign["mark"],        placeholder="e.g. Extreme breathing difficulty", label_visibility="collapsed", key=f"wm_{i}")
        sign["instruction"] = c2.text_input("p", value=sign["instruction"], placeholder="e.g. Go to emergency immediately",  label_visibility="collapsed", key=f"wi_{i}")
        if c3.button(":material/delete:", key=f"wd_{i}"):
            to_del_w = i
    if to_del_w is not None:
        st.session_state.warning_signs.pop(to_del_w)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── D. Emergency Handover Plan ────────────────────────────────────────────
    st.markdown("<div class='tx-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='tx-section-title'>:material/local_hospital: &nbsp;Emergency Handover Plan (Night Shift)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#1e90ff; font-size:0.85rem; margin-bottom:0.8rem;'>"
        "Critical instructions for on-call emergency physicians.</p>",
        unsafe_allow_html=True,
    )
    st.text_area("Critical Patient Risks",          placeholder="Potential respiratory failure, specific allergies, cardiac risk…",    height=85, key="ho_risks")
    st.text_area("Immediate Actions to Take",       placeholder="Oxygen therapy settings, emergency medications…",                     height=85, key="ho_actions")
    st.text_area("What Emergency Doctor Must Know", placeholder="Recent clinical changes, guardian contacts, known allergies…",         height=85, key="ho_must_know")
    st.text_area("Medical History Summary",         placeholder="Chronic conditions relevant to acute care: diabetes, heart disease…",  height=85, key="ho_history")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
    if st.button(
        ":material/save: Save Treatment Plan",
        use_container_width=True,
        key="save_home",
        type="primary",
    ):
        st.success(":material/check_circle: Home treatment plan saved successfully. (Placeholder)")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (defined at module level so they work with forward refs)
# ─────────────────────────────────────────────────────────────────────────────

def _default_check_rows() -> list[dict]:
    """Return the WHO-aligned default monitoring rows for one check table."""
    rows = [
        {"#": 1,  "Feature": "Temperature",        "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 2,  "Feature": "Respiratory Rate",    "Current Status": "", "vs. Last Check": "", "Notes": "breaths/min"},
        {"#": 3,  "Feature": "Oxygen Saturation",   "Current Status": "", "vs. Last Check": "", "Notes": "Alert if <90%"},
        {"#": 4,  "Feature": "Chest Indrawing",     "Current Status": "", "vs. Last Check": "", "Notes": "WHO danger sign"},
        {"#": 5,  "Feature": "Work of Breathing",   "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 6,  "Feature": "Consciousness",       "Current Status": "", "vs. Last Check": "", "Notes": "Any ↓ = CRITICAL"},
        {"#": 7,  "Feature": "Feeding / Drinking",  "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 8,  "Feature": "Chest Pain",          "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 9,  "Feature": "Crackles",            "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 10, "Feature": "Breath Sounds",       "Current Status": "", "vs. Last Check": "", "Notes": ""},
        {"#": 11, "Feature": "Cyanosis",            "Current Status": "", "vs. Last Check": "", "Notes": "Central = emergency"},
        {"#": 12, "Feature": "Antibiotic Dose Given","Current Status": "", "vs. Last Check": "", "Notes": "Track compliance"},
        {"#": 13, "Feature": "Hours Since Last Fever","Current Status": "", "vs. Last Check": "", "Notes": "0 = currently febrile"},
        {"#": 14, "Feature": "Clinical Impression", "Current Status": "", "vs. Last Check": "", "Notes": "Nurse assessment"},
        {"#": 15, "Feature": "Red Flags Present",   "Current Status": "", "vs. Last Check": "", "Notes": "Count total"},
    ]
    # Seed extra column fields
    for row in rows:
        row["extra"] = {col: "" for col in st.session_state.get("extra_columns", [])}
    return rows


def _render_check_tables():
    """Render all accumulated check tables with editable nurse inputs."""
    if not st.session_state.check_tables:
        st.markdown(
            "<p style='color:#7a8fa8; font-size:0.88rem; margin-top:0.6rem;'>"
            ":material/table_chart: &nbsp;No check tables yet — start monitoring to generate the first table.</p>",
            unsafe_allow_html=True,
        )
        return

    base_cols   = ["#", "Feature", "Current Status", "vs. Last Check", "Notes"]
    extra_cols  = st.session_state.extra_columns
    all_cols    = base_cols + extra_cols

    for tbl in st.session_state.check_tables:
        tidx = tbl["index"]
        rows = tbl["rows"]

        # ── Table title bar ───────────────────────────────────────────────────
        st.markdown(
            f"<div class='check-table-header'>"
            f":material/table_chart: &nbsp;Check Table {tidx} &nbsp;|&nbsp; Table de suivi</div>",
            unsafe_allow_html=True,
        )

        # ── Build editable Streamlit columns grid ────────────────────────────
        # Header row
        widths = [0.3, 1.8, 2, 1.5, 1.5] + [1.5] * len(extra_cols)
        hdr = st.columns(widths)
        for col, lbl in zip(hdr, all_cols):
            col.markdown(
                f"<div style='background:#e8eef8; color:#001f3f; font-weight:700; "
                f"font-size:0.78rem; padding:0.3rem 0.4rem; border-bottom:2px solid #001f3f;'>"
                f"{lbl}</div>",
                unsafe_allow_html=True,
            )

        # Data rows
        for row in rows:
            cols = st.columns(widths)
            ri   = row["#"]

            cols[0].markdown(
                f"<div style='font-size:0.8rem; padding:0.25rem 0.4rem; color:#001f3f; font-weight:600;'>{ri}</div>",
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f"<div style='font-size:0.8rem; padding:0.25rem 0.4rem;'>{row['Feature']}</div>",
                unsafe_allow_html=True,
            )

            row["Current Status"] = cols[2].text_input(
                "cs", value=row["Current Status"],
                placeholder="Observation…",
                label_visibility="collapsed",
                key=f"t{tidx}_r{ri}_cs",
            )
            row["vs. Last Check"] = cols[3].text_input(
                "lc", value=row["vs. Last Check"],
                placeholder="↓ → ↑",
                label_visibility="collapsed",
                key=f"t{tidx}_r{ri}_lc",
            )
            notes_hint = row.get("Notes", "")
            cols[4].markdown(
                f"<div style='font-size:0.75rem; color:#5a7a9a; padding:0.3rem 0.2rem;'>{notes_hint}</div>",
                unsafe_allow_html=True,
            )

            # Extra doctor-defined columns
            for ci, ecol in enumerate(extra_cols):
                row["extra"] = row.get("extra", {})
                row["extra"][ecol] = cols[5 + ci].text_input(
                    "ec", value=row["extra"].get(ecol, ""),
                    placeholder="—",
                    label_visibility="collapsed",
                    key=f"t{tidx}_r{ri}_ex{ci}",
                )

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)


# ── Back button ──────────────────────────────────────────────────────────────
st.markdown("<div class='tx-divider'></div>", unsafe_allow_html=True)
if st.button(":material/arrow_back: Back to Home"):
    st.switch_page("app.py")