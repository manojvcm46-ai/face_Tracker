"""
Root launcher proxy for app.py.
Allows running `streamlit run app.py` directly from the workspace root.
"""
import os
import sys

# Ensure current working directory and sys.path point to ai_face_recognition
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_face_recognition")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(project_dir)

# Run main application
with open(os.path.join(project_dir, "app.py"), "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, os.path.join(project_dir, "app.py"), "exec"))
