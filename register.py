"""
Root launcher proxy for register.py.
Allows running `python register.py` directly from the workspace root.
"""
import os
import sys

# Ensure current working directory and sys.path point to ai_face_recognition
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_face_recognition")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.chdir(project_dir)

from register import run_cli_registration

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enroll a new employee face into SQLite database.")
    parser.add_argument("--id", type=str, default="", help="Employee ID (e.g. EMP101)")
    parser.add_argument("--name", type=str, default="", help="Full Name")
    parser.add_argument("--role", type=str, default="Staff", help="Role / Designation")
    parser.add_argument("--dept", type=str, default="General", help="Department")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--samples", type=int, default=3, help="Number of face samples to register")

    args = parser.parse_args()

    emp_id = args.id or input("Enter Employee ID: ").strip()
    emp_name = args.name or input("Enter Full Name: ").strip()
    emp_role = args.role if args.id else input("Enter Role / Designation: ").strip() or "Staff"
    emp_dept = args.dept if args.id else input("Enter Department: ").strip() or "General"

    run_cli_registration(
        employee_id=emp_id,
        name=emp_name,
        role=emp_role,
        department=emp_dept,
        camera_index=args.camera,
        samples_target=args.samples
    )
