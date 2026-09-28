"""
Streamlit Web Application: AI Face Recognition Command Center.
Modern professional dark-themed biometric recognition dashboard with live webcam preview,
employee enrollment, real-time matching, SQLite database management, and audit logs.
"""

import os
import sys
import time
import warnings

# Suppress Streamlit deprecation warnings for clean performance
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and visual styling
st.set_page_config(
    page_title="AI Face Recognition | VisionGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from database import db
from face_engine import FaceEngine
from register import process_registration_frame, register_single_face
from recognize import FaceRecognizerPipeline

# Custom Glassmorphic Dark-Themed Styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Glow */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(13, 23, 42, 1) 0%, rgba(8, 12, 22, 1) 90%);
        color: #e2e8f0;
    }

    /* Top Header Card */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Status Pill / Live Badge */
    .status-badge-active {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .status-badge-inactive {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #f87171;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.25); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Metric Stat Card */
    .stat-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
    }

    .stat-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Profile Side Floating Card */
    .profile-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(74, 222, 128, 0.3);
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
    }

    .profile-card-unknown {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .biometric-notice {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #fbbf24;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.45;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "face_engine" not in st.session_state:
    st.session_state.face_engine = FaceEngine()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = FaceRecognizerPipeline(engine=st.session_state.face_engine)


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    camera_idx = st.number_input("Webcam Hardware Index", min_value=0, max_value=5, value=0, step=1)
    
    threshold = st.slider(
        "Matching Strictness (Cosine Threshold)",
        min_value=0.25,
        max_value=0.55,
        value=0.363,
        step=0.01,
        help="Higher value requires closer match. OpenCV SFace standard benchmark is 0.363."
    )
    st.session_state.face_engine.cosine_threshold = threshold

    detection_score_th = st.slider(
        "Face Detector Confidence (YuNet)",
        min_value=0.40,
        max_value=0.90,
        value=0.70,
        step=0.05
    )
    st.session_state.face_engine.score_threshold = detection_score_th

    st.markdown("---")
    st.markdown("### 📊 Database Overview")
    stats = db.get_database_stats()
    st.metric("Enrolled Employees", stats["total_employees"])
    st.metric("Stored Embeddings", stats["total_embeddings"])
    st.metric("Today's Verifications", stats["today_present"])

    if st.button("🔄 Reload Database Cache", use_container_width=True):
        st.session_state.pipeline.reload_enrolled_faces()
        st.success("Database cache refreshed!")

    st.markdown("---")
    st.markdown(
        """
        <div class="biometric-notice">
            🔒 <b>Privacy Architecture:</b> Zero raw image retention. Only 128-D mathematical feature embeddings are securely stored locally in SQLite.
        </div>
        """,
        unsafe_allow_html=True
    )


# Top Hero Header
st.markdown("""
<div class="hero-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 class="hero-title">
                <span>🛡️ VisionGuard AI</span>
            </h1>
            <div class="hero-subtitle">Production-Ready Biometric Face Recognition & Identity Verification Engine</div>
        </div>
        <div id="live-status-container">
            <div class="status-badge-active">
                <span class="pulse-dot"></span>
                <span>AI Face Recognition Active</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Modern Top Navigation Bar
nav_options = [
    "🎯 Real-Time Recognition",
    "📝 Register Face",
    "👥 Employee Directory",
    "📜 Attendance & Audit Logs"
]

app_mode = st.radio(
    "Navigation Menu",
    nav_options,
    horizontal=True,
    label_visibility="collapsed"
)

# Critical: Automatically stop and release the camera whenever switching away from Real-Time Recognition
if app_mode != "🎯 Real-Time Recognition":
    st.session_state.camera_active = False

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: REAL-TIME RECOGNITION
# ==============================================================================
if app_mode == "🎯 Real-Time Recognition":
    col_video, col_side = st.columns([2.8, 1.2])

    with col_side:
        st.markdown("### 🎮 Control Center")
        
        btn_start = st.button("▶️ Start Recognition Camera", type="primary", key="btn_start", use_container_width=True)
        btn_stop = st.button("⏹️ Stop Camera", type="secondary", key="btn_stop", use_container_width=True)

        if btn_start:
            st.session_state.camera_active = True
            st.rerun()
        if btn_stop:
            st.session_state.camera_active = False
            st.rerun()

        st.markdown("---")
        st.markdown("### 📡 Live Telemetry")
        
        status_box = st.empty()
        faces_box = st.empty()
        fps_box = st.empty()
        profile_feed = st.empty()

        if st.session_state.camera_active:
            status_box.markdown(
                '<div class="status-badge-active"><span class="pulse-dot"></span> Camera: Streaming (Index %d)</div>' % camera_idx,
                unsafe_allow_html=True
            )
        else:
            status_box.markdown(
                '<div class="status-badge-inactive">⭕ Camera: Inactive / Stopped</div>',
                unsafe_allow_html=True
            )
            faces_box.info("Detected Faces: 0")
            fps_box.text("Frame Rate: 0.0 FPS")

    with col_video:
        video_placeholder = st.empty()

        if st.session_state.camera_active:
            cap = None
            try:
                # On Windows, DirectShow (CAP_DSHOW) is fast and reliable; fallback to default if needed
                cap = cv2.VideoCapture(int(camera_idx), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(camera_idx))

                # Verify that frames can actually be grabbed
                ret, test_frame = cap.read() if cap.isOpened() else (False, None)
                if not ret or test_frame is None:
                    if cap is not None and cap.isOpened():
                        cap.release()
                    # Try alternate backend
                    cap = cv2.VideoCapture(int(camera_idx))
                    ret, test_frame = cap.read() if cap.isOpened() else (False, None)

                if not cap.isOpened() or not ret or test_frame is None:
                    st.error(
                        f"❌ Cannot access webcam #{camera_idx}. The camera may be locked by another application "
                        "(such as a running terminal session, Zoom, or Teams). "
                        "Please close other camera windows and try again."
                    )
                    st.session_state.camera_active = False
                else:
                    # Safely set dimensions if supported by camera driver
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    except Exception:
                        pass

                    pipeline = st.session_state.pipeline
                    pipeline.reload_enrolled_faces()

                    prev_time = time.time()
                    fps = 0.0
                    consecutive_drops = 0
                    frame_idx = 0

                    # Real-time frame loop
                    while st.session_state.get("camera_active", False):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_drops += 1
                            if consecutive_drops > 25:
                                st.warning("Webcam feed disconnected.")
                                break
                            time.sleep(0.02)
                            continue
                        consecutive_drops = 0
                        frame_idx += 1

                        # FPS Calculation
                        now = time.time()
                        fps = 0.85 * fps + 0.15 * (1.0 / max(0.001, (now - prev_time)))
                        prev_time = now

                        # Check if camera sensor is pitch black (physical privacy shutter closed)
                        is_black_screen = frame.mean() < 3.0

                        # Process frame through Face Engine
                        annotated, results, face_count = pipeline.process_frame(frame, draw_overlays=True)

                        # If screen is pitch black, display helpful on-screen guidance
                        if is_black_screen:
                            h_f, w_f = annotated.shape[:2]
                            cv2.rectangle(annotated, (30, h_f // 2 - 45), (w_f - 30, h_f // 2 + 45), (20, 24, 32), -1)
                            cv2.rectangle(annotated, (30, h_f // 2 - 45), (w_f - 30, h_f // 2 + 45), (0, 165, 255), 2)
                            cv2.putText(
                                annotated,
                                "! CAMERA SHUTTER CLOSED OR COVERED",
                                (45, h_f // 2 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 200, 255),
                                2,
                                cv2.LINE_AA
                            )
                            cv2.putText(
                                annotated,
                                "Please open the physical privacy slider / switch on your webcam.",
                                (45, h_f // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (220, 225, 235),
                                1,
                                cv2.LINE_AA
                            )

                        # Throttle Streamlit DOM markdown telemetry updates to every 8 frames to avoid WebSocket lag
                        if frame_idx % 8 == 0:
                            faces_box.markdown(f"**Detected Faces:** `{face_count}`")
                            fps_box.markdown(f"**Tracking Speed:** `{fps:.1f} FPS`")

                            if is_black_screen:
                                profile_feed.warning("Camera physical shutter is closed or sensor is receiving zero light.")
                            elif results:
                                cards_html = ""
                                for res in results:
                                    if res["is_recognized"] and res["employee"]:
                                        emp = res["employee"]
                                        cards_html += f"""
                                        <div class="profile-card">
                                            <div style="font-size:0.75rem; color:#4ade80; font-weight:700;">● VERIFIED EMPLOYEE</div>
                                            <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin: 4px 0;">{emp['name']}</div>
                                            <div style="font-size:0.85rem; color:#cbd5e1;"><b>ID:</b> {emp['employee_id']}</div>
                                            <div style="font-size:0.85rem; color:#94a3b8;"><b>Role:</b> {emp['role']}</div>
                                            <div style="font-size:0.85rem; color:#94a3b8;"><b>Dept:</b> {emp['department']}</div>
                                            <div style="margin-top:8px; font-size:0.85rem; color:#4ade80;"><b>Match Confidence:</b> {res['confidence']:.1f}%</div>
                                        </div>
                                        """
                                    else:
                                        cards_html += f"""
                                        <div class="profile-card-unknown">
                                            <div style="font-size:0.75rem; color:#f87171; font-weight:700;">⚠️ SECURITY ALERT</div>
                                            <div style="font-size:1.15rem; font-weight:700; color:#ffffff; margin: 4px 0;">Unknown Person</div>
                                            <div style="font-size:0.85rem; color:#fca5a5;">Access Denied / Unregistered</div>
                                            <div style="font-size:0.80rem; color:#94a3b8; margin-top:4px;">No matching biometric signature found.</div>
                                        </div>
                                        """
                                profile_feed.markdown(cards_html, unsafe_allow_html=True)
                            else:
                                profile_feed.info("Position face within camera view.")

                        # Convert BGR to RGB for Streamlit rendering
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB")

                        # Small yield to keep UI responsive
                        time.sleep(0.02)

            except Exception as cam_err:
                st.error(f"Camera stream error: {cam_err}. Please ensure webcam is not in use elsewhere.")
                st.session_state.camera_active = False

            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                st.session_state.camera_active = False
        else:
            # Idle placeholder illustration
            video_placeholder.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(255, 255, 255, 0.15); border-radius: 14px; padding: 100px 20px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 12px;">📷</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #cbd5e1;">Camera Stream Inactive</div>
                <div style="font-size: 0.9rem; color: #64748b; margin-top: 6px;">Click <b>Start Recognition Camera</b> to initiate real-time AI biometric tracking</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 2: REGISTER FACE (FIRST TIME REGISTRATION)
# ==============================================================================
elif app_mode == "📝 Register Face":
    st.markdown("### 📝 New Employee Face Registration")
    st.write("Enroll employee biometric profile. Exactly one person must be visible in the image during capture.")

    col_form, col_cam = st.columns([1.1, 1.9])

    with col_form:
        st.markdown("#### Employee Information")
        with st.form("employee_registration_form"):
            reg_id = st.text_input("Employee ID *", placeholder="e.g. EMP1001").strip()
            reg_name = st.text_input("Full Name *", placeholder="e.g. John Doe").strip()
            reg_role = st.text_input("Role / Designation *", placeholder="e.g. Senior ML Engineer").strip()
            reg_dept = st.text_input("Department *", placeholder="e.g. Artificial Intelligence").strip()
            
            samples_to_take = st.select_slider(
                "Samples to Enroll (Recommended: 3)",
                options=[1, 2, 3, 5],
                value=3,
                help="Multiple samples from slightly different angles/lighting improve recognition accuracy."
            )

            consent_agreed = st.checkbox(
                "User has provided biometric enrollment consent",
                value=True,
                help="Mandatory biometric consent compliance statement."
            )

            submit_form = st.form_submit_button("Lock Details & Proceed to Capture", type="primary", use_container_width=True)

    with col_cam:
        st.markdown("#### Image Capture & Biometric Extraction")

        reg_source = st.radio(
            "Enrollment Mode",
            ["📷 Browser Camera Snapshot", "📁 Upload Photo File", "🖥️ Desktop CLI Camera"],
            horizontal=True
        )

        cv_img = None

        if reg_source == "📷 Browser Camera Snapshot":
            camera_img = st.camera_input("Take Enrollment Snapshot")
            if camera_img is not None:
                bytes_data = camera_img.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        elif reg_source == "📁 Upload Photo File":
            uploaded_file = st.file_uploader("Upload Person Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        else:
            st.info("Direct Desktop Window Mode: Runs a standalone OpenCV camera window.")
            st.code(
                f'python register.py --id "{reg_id or "EMP001"}" --name "{reg_name or "Jane Doe"}" --role "{reg_role or "Engineer"}" --dept "{reg_dept or "AI"}" --samples {samples_to_take}',
                language="bash"
            )

        # If an image has been provided via Snapshot or File Upload
        if cv_img is not None:
            # Validate frame with engine
            annotated_reg, count, primary_face, msg = process_registration_frame(
                cv_img, st.session_state.face_engine
            )

            if count == 1 and primary_face is not None:
                st.success(f"✅ {msg}")
                st.image(cv2.cvtColor(annotated_reg, cv2.COLOR_BGR2RGB), channels="RGB")

                if st.button("💾 Enroll Face to Database", type="primary", use_container_width=True):
                    if not reg_id or not reg_name:
                        st.error("Please fill in Employee ID and Name in the form on the left.")
                    elif not consent_agreed:
                        st.error("Biometric consent must be confirmed before enrollment.")
                    else:
                        success, ret_msg = register_single_face(
                            image=cv_img,
                            employee_id=reg_id,
                            name=reg_name,
                            role=reg_role or "Staff",
                            department=reg_dept or "General",
                            engine=st.session_state.face_engine
                        )
                        if success:
                            st.balloons()
                            st.success(f"🎉 {ret_msg}")
                            st.session_state.pipeline.reload_enrolled_faces()
                        else:
                            st.error(ret_msg)
            elif count == 0:
                st.warning("⚠️ No face detected in this image. Please ensure the face is clearly visible and well lit.")
                st.image(cv2.cvtColor(annotated_reg, cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                st.error("⚠️ Multiple faces detected! Exactly one person is required for registration.")
                st.image(cv2.cvtColor(annotated_reg, cv2.COLOR_BGR2RGB), channels="RGB")


# ==============================================================================
# SECTION 3: EMPLOYEE DIRECTORY & DATABASE MANAGEMENT
# ==============================================================================
elif app_mode == "👥 Employee Directory":
    st.markdown("### 👥 Registered Employees Directory")
    employees = db.get_all_employees()

    if employees:
        df_emp = pd.DataFrame(employees)
        # Select and rename columns for display
        df_display = df_emp[["employee_id", "name", "role", "department", "sample_count", "created_at"]]
        df_display.columns = ["Employee ID", "Full Name", "Role", "Department", "Face Samples", "Enrolled Date"]

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("#### 🗑️ Employee Record Management")
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            emp_to_delete = st.selectbox(
                "Select Employee to Remove",
                options=[f"{e['employee_id']} - {e['name']}" for e in employees]
            )
        with del_col2:
            st.write("")
            st.write("")
            if st.button("Delete Employee", type="secondary", use_container_width=True):
                target_id = emp_to_delete.split(" - ")[0]
                if db.delete_employee(target_id):
                    st.success(f"Removed employee {target_id} and all biometric embeddings.")
                    st.session_state.pipeline.reload_enrolled_faces()
                    st.rerun()
                else:
                    st.error("Failed to delete employee.")
    else:
        st.info("No employees registered in database yet. Go to 'Register Face' menu to enroll your first employee.")


# ==============================================================================
# SECTION 4: ATTENDANCE & AUDIT LOGS
# ==============================================================================
elif app_mode == "📜 Attendance & Audit Logs":
    st.markdown("### 📜 Real-Time Verification & Attendance Logs")
    logs = db.get_attendance_logs(limit=100)

    if logs:
        df_logs = pd.DataFrame(logs)
        df_logs_display = df_logs[["timestamp", "employee_id", "name", "role", "department", "confidence", "status"]]
        df_logs_display.columns = ["Timestamp", "Employee ID", "Name", "Role", "Department", "Confidence %", "Status"]

        st.dataframe(df_logs_display, use_container_width=True, hide_index=True)

        # Download CSV export option
        csv_data = df_logs_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs as CSV",
            data=csv_data,
            file_name="attendance_verification_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No verification logs recorded yet. Recognized faces will automatically log here.")
