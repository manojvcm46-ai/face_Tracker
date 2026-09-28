"""
Database Management Module for AI Face Recognition System.
Handles SQLite storage and retrieval of employee records, face embeddings,
and audit/attendance activity logs with privacy-by-design standards.
"""

import os
import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Resolve default database file path inside project data directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(CURRENT_DIR, "data", "face_database.db")


class FaceDatabase:
    """SQLite Database manager for face embeddings and employee metadata."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable Foreign Key enforcement in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables with schema constraints and indexing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Employees Metadata Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Face Embeddings Table (128-D vector stored as serialized float32 binary)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER DEFAULT 128,
                    quality_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
                );
            """)

            # 3. Attendance & Access Activity Logs Table (Future-ready architecture)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'VERIFIED',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
                );
            """)

            # Create Indexes for fast querying during real-time matching
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emp_id ON employees(employee_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_emp ON face_embeddings(employee_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_emp ON attendance_logs(employee_id);")
            conn.commit()

    @staticmethod
    def embedding_to_blob(embedding: np.ndarray) -> bytes:
        """Convert a 1D/2D float32 numpy embedding array to raw bytes."""
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        return arr.tobytes()

    @staticmethod
    def blob_to_embedding(blob: bytes, dim: int = 128) -> np.ndarray:
        """Convert raw bytes back to a 1D float32 numpy embedding array."""
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0 and len(arr) != dim:
            # SFace embeddings are 128 dimensions; dlib embeddings are also 128 dimensions
            return arr.reshape(-1)
        return arr

    def add_employee(self, employee_id: str, name: str, role: str, department: str) -> bool:
        """
        Insert or update employee details in the database.
        Returns True if successful, False otherwise.
        """
        employee_id = employee_id.strip()
        name = name.strip()
        role = role.strip()
        department = department.strip()

        if not employee_id or not name:
            raise ValueError("Employee ID and Name cannot be empty.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO employees (employee_id, name, role, department, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(employee_id) DO UPDATE SET
                    name = excluded.name,
                    role = excluded.role,
                    department = excluded.department,
                    updated_at = CURRENT_TIMESTAMP;
            """, (employee_id, name, role, department))
            conn.commit()
            return True

    def add_face_embedding(self, employee_id: str, embedding: np.ndarray, quality_score: float = 1.0) -> int:
        """
        Save a face embedding vector for an employee.
        An employee can have multiple sample embeddings to improve accuracy across lighting/angles.
        """
        employee_id = employee_id.strip()
        blob = self.embedding_to_blob(embedding)
        dim = len(np.asarray(embedding, dtype=np.float32).flatten())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO face_embeddings (employee_id, embedding, embedding_dim, quality_score)
                VALUES (?, ?, ?, ?);
            """, (employee_id, blob, dim, quality_score))
            conn.commit()
            return cursor.lastrowid

    def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Fetch employee metadata by employee_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT employee_id, name, role, department, created_at, updated_at
                FROM employees
                WHERE employee_id = ?;
            """, (employee_id.strip(),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_employees(self) -> List[Dict[str, Any]]:
        """Fetch list of all registered employees with sample counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    e.employee_id,
                    e.name,
                    e.role,
                    e.department,
                    e.created_at,
                    e.updated_at,
                    COUNT(f.id) as sample_count
                FROM employees e
                LEFT JOIN face_embeddings f ON e.employee_id = f.employee_id
                GROUP BY e.employee_id
                ORDER BY e.created_at DESC;
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_all_embeddings_with_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve all stored embeddings along with full employee metadata.
        Used for fast in-memory matching against camera feeds.
        Returns:
            List of dicts: [
                {
                    'employee_id': 'EMP001',
                    'name': 'Jane Doe',
                    'role': 'AI Engineer',
                    'department': 'R&D',
                    'embedding': np.ndarray,
                    'dim': 128
                }, ...
            ]
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    e.employee_id,
                    e.name,
                    e.role,
                    e.department,
                    f.embedding,
                    f.embedding_dim
                FROM face_embeddings f
                JOIN employees e ON f.employee_id = e.employee_id;
            """)
            records = []
            for row in cursor.fetchall():
                emb_arr = self.blob_to_embedding(row["embedding"], row["embedding_dim"])
                records.append({
                    "employee_id": row["employee_id"],
                    "name": row["name"],
                    "role": row["role"],
                    "department": row["department"],
                    "embedding": emb_arr,
                    "dim": row["embedding_dim"]
                })
            return records

    def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee and all associated face embeddings and logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM employees WHERE employee_id = ?;", (employee_id.strip(),))
            conn.commit()
            return cursor.rowcount > 0

    def delete_embeddings_for_employee(self, employee_id: str) -> int:
        """Clear face embeddings for an employee while keeping their record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings WHERE employee_id = ?;", (employee_id.strip(),))
            conn.commit()
            return cursor.rowcount

    def log_attendance(self, employee_id: str, confidence: float, status: str = "VERIFIED") -> int:
        """
        Log an entry for a recognized employee.
        Prevents rapid duplicate logs if logged within the last 30 seconds.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Debounce check: don't log if logged within past 30 seconds
            cursor.execute("""
                SELECT id FROM attendance_logs
                WHERE employee_id = ? AND timestamp >= datetime('now', '-30 seconds')
                ORDER BY timestamp DESC LIMIT 1;
            """, (employee_id,))
            recent = cursor.fetchone()
            if recent:
                return recent["id"]

            cursor.execute("""
                INSERT INTO attendance_logs (employee_id, confidence, status)
                VALUES (?, ?, ?);
            """, (employee_id, float(confidence), status))
            conn.commit()
            return cursor.lastrowid

    def get_attendance_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent attendance/recognition logs joined with employee names."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    a.id,
                    a.employee_id,
                    e.name,
                    e.role,
                    e.department,
                    a.confidence,
                    a.status,
                    a.timestamp
                FROM attendance_logs a
                LEFT JOIN employees e ON a.employee_id = e.employee_id
                ORDER BY a.timestamp DESC
                LIMIT ?;
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_database_stats(self) -> Dict[str, Any]:
        """Return system statistics for dashboard status cards."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM employees;")
            total_employees = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM face_embeddings;")
            total_embeddings = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM attendance_logs;")
            total_logs = cursor.fetchone()["count"]

            cursor.execute("""
                SELECT COUNT(DISTINCT employee_id) as count 
                FROM attendance_logs 
                WHERE date(timestamp) = date('now');
            """)
            today_present = cursor.fetchone()["count"]

            return {
                "total_employees": total_employees,
                "total_embeddings": total_embeddings,
                "total_logs": total_logs,
                "today_present": today_present
            }


# Singleton instance for convenient reuse
db = FaceDatabase()
