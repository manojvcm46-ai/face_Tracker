"""
Root launcher proxy for recognize.py.
Allows running `python recognize.py` directly from the workspace root.
"""
import os
import sys

# Ensure current working directory and sys.path point to ai_face_recognition
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_face_recognition")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(project_dir)

from recognize import run_standalone_recognition

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Real-time Face Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.363, help="Cosine threshold (default: 0.363)")
    args = parser.parse_args()

    run_standalone_recognition(camera_index=args.camera, conf_threshold=args.threshold)
