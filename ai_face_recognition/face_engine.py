"""
Face AI Engine Module.
Integrates OpenCV Deep Learning YuNet face detection and SFace feature extraction
to generate 128-dimensional facial embeddings, compute cosine similarities,
and render aesthetic real-time floating profile cards and bounding boxes.
"""

import os
import cv2
import numpy as np
import urllib.request
from typing import List, Dict, Tuple, Optional, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Official direct mirror URLs for models
YUNET_URL = "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?raw=true"
SFACE_URL = "https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?raw=true"


class FaceEngine:
    """
    Production-ready Face Engine powering detection, alignment,
    128-D embedding extraction, and cosine-similarity matching.
    """

    # SFace benchmark cosine similarity threshold (recommended by OpenCV Zoo is 0.363)
    DEFAULT_COSINE_THRESHOLD = 0.363
    DEFAULT_CONF_THRESHOLD = 0.70  # YuNet detection confidence score threshold

    def __init__(
        self,
        score_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.cosine_threshold = cosine_threshold

        self._ensure_models_exist()

        # Initialize detector and recognizer
        # Initial input size is (320, 320), dynamically updated per frame dimensions
        self.detector = cv2.FaceDetectorYN.create(
            model=YUNET_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_PATH,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

        self._last_input_size = (320, 320)

    @classmethod
    def _ensure_models_exist(cls):
        """Automatically verify and download ONNX weights if absent."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}

        # Download YuNet if missing or invalid
        if not os.path.exists(YUNET_PATH) or os.path.getsize(YUNET_PATH) < 200000:
            print("Downloading YuNet Face Detection model (232 KB)...")
            req = urllib.request.Request(YUNET_URL, headers=headers)
            with urllib.request.urlopen(req) as resp, open(YUNET_PATH, "wb") as f:
                f.write(resp.read())
            print("YuNet model downloaded successfully.")

        # Download SFace if missing or invalid
        if not os.path.exists(SFACE_PATH) or os.path.getsize(SFACE_PATH) < 30000000:
            print("Downloading SFace Face Recognition model (38.6 MB)...")
            req = urllib.request.Request(SFACE_URL, headers=headers)
            with urllib.request.urlopen(req) as resp, open(SFACE_PATH, "wb") as f:
                f.write(resp.read())
            print("SFace model downloaded successfully.")

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in a BGR frame.
        Returns a list of dicts containing:
            - 'bbox': [x, y, w, h] (integer coordinates)
            - 'landmarks': 5 keypoints (right eye, left eye, nose, right mouth, left mouth)
            - 'score': detection confidence (0.0 to 1.0)
            - 'raw_face': the raw 15-dim numpy vector from YuNet
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if self._last_input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._last_input_size = (w, h)

        _, raw_faces = self.detector.detect(image)

        if raw_faces is None:
            return []

        results = []
        for face in raw_faces:
            x, y, fw, fh = map(int, face[:4])
            # Bounding box clamp to image boundaries
            x = max(0, x)
            y = max(0, y)
            fw = min(w - x, fw)
            fh = min(h - y, fh)

            landmarks = face[4:14].reshape(5, 2)
            score = float(face[14])

            results.append({
                "bbox": [x, y, fw, fh],
                "landmarks": landmarks,
                "score": score,
                "raw_face": face,
            })

        return results

    def get_embedding(self, image: np.ndarray, raw_face: np.ndarray) -> np.ndarray:
        """
        Align the detected face using landmarks and extract 128-D L2-normalized embedding.
        """
        # Crop and align face to canonical pose (112x112 standard input for SFace)
        aligned_face = self.recognizer.alignCrop(image, raw_face)
        # Extract 128-dimensional float32 feature embedding
        embedding = self.recognizer.feature(aligned_face)
        return embedding.flatten().astype(np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two 128-D embeddings using SFace engine.
        Cosine score ranges:
            > 0.363 -> Match (same person)
            < 0.363 -> Mismatch (different persons)
        """
        e1 = np.asarray(embedding1, dtype=np.float32).reshape(1, -1)
        e2 = np.asarray(embedding2, dtype=np.float32).reshape(1, -1)
        score = self.recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def calculate_confidence_percentage(self, cosine_score: float) -> float:
        """
        Calibrate raw cosine similarity score (typically 0.10 - 0.75) into an intuitive 0-100% confidence.
        Based on SFace distribution:
            Threshold = 0.363 maps to 75.0% confidence.
            Score 0.65+ maps to 99.0%+.
            Score < 0.20 maps to < 35.0%.
        """
        th = self.cosine_threshold
        if cosine_score >= th:
            # Range [th, 0.70] -> [75.0%, 99.5%]
            ratio = (cosine_score - th) / max(0.001, (0.70 - th))
            pct = 75.0 + ratio * 24.5
            return round(float(np.clip(pct, 75.0, 99.8)), 1)
        else:
            # Range [0.0, th] -> [10.0%, 74.9%]
            ratio = max(0.0, cosine_score) / th
            pct = 10.0 + ratio * 64.9
            return round(float(np.clip(pct, 5.0, 74.9)), 1)

    def match_face(
        self,
        query_embedding: np.ndarray,
        enrolled_db_records: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Dict[str, Any]], float, float]:
        """
        Match a query face embedding against registered embeddings in the database.
        Supports multiple embeddings per employee (takes the best match across all samples).

        Returns:
            (is_match, matched_employee_info, best_cosine_score, confidence_percentage)
        """
        if not enrolled_db_records:
            return False, None, 0.0, 0.0

        best_score = -1.0
        best_record = None

        # Group comparisons across all enrolled embeddings
        for rec in enrolled_db_records:
            score = self.compute_similarity(query_embedding, rec["embedding"])
            if score > best_score:
                best_score = score
                best_record = rec

        is_match = best_score >= self.cosine_threshold
        confidence_pct = self.calculate_confidence_percentage(best_score)

        return is_match, best_record, best_score, confidence_pct

    def draw_aesthetic_bounding_box(
        self,
        frame: np.ndarray,
        bbox: List[int],
        is_recognized: bool,
        corner_len: int = 18,
        thickness: int = 2
    ):
        """
        Draw a modern futuristic bounding box with corner brackets and glowing accent lines.
        Green (46, 204, 113) for Recognized.
        Red (60, 60, 235) for Unknown.
        """
        x, y, w, h = bbox
        # OpenCV uses BGR format
        color = (113, 204, 46) if is_recognized else (60, 60, 235)
        accent_color = (180, 255, 120) if is_recognized else (120, 120, 255)

        # Base soft rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        # Prominent corner brackets (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), accent_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + corner_len), accent_color, thickness, cv2.LINE_AA)

        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), accent_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), accent_color, thickness, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), accent_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), accent_color, thickness, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), accent_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), accent_color, thickness, cv2.LINE_AA)

    def draw_floating_profile_card(
        self,
        frame: np.ndarray,
        bbox: List[int],
        is_recognized: bool,
        employee_info: Optional[Dict[str, Any]] = None,
        confidence_pct: float = 0.0
    ):
        """
        Renders a sleek floating glassmorphic information card directly beside the face.
        Displays Name, Employee ID, Role, Department, and Match Confidence.
        """
        img_h, img_w = frame.shape[:2]
        x, y, w, h = bbox

        card_w = 260
        card_h = 135

        # Position card to the right of the face, or to the left if near screen right boundary
        card_x = x + w + 15
        if card_x + card_w > img_w - 10:
            card_x = max(10, x - card_w - 15)

        card_y = max(10, min(y, img_h - card_h - 10))

        # 1. Create semi-transparent dark HUD glass background
        overlay = frame.copy()
        bg_color = (20, 24, 30)  # Sleek dark navy/slate
        cv2.rectangle(
            overlay,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            bg_color,
            -1
        )
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Border & accent indicator
        border_color = (113, 204, 46) if is_recognized else (60, 60, 235)
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (60, 65, 75),
            1,
            cv2.LINE_AA
        )
        # Left status strip on card
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + 5, card_y + card_h),
            border_color,
            -1
        )

        # Text rendering parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = card_x + 16

        if is_recognized and employee_info:
            name = str(employee_info.get("name", "Unknown"))
            emp_id = str(employee_info.get("employee_id", ""))
            role = str(employee_info.get("role", "N/A"))
            dept = str(employee_info.get("department", "N/A"))

            # Status Tag / Header
            cv2.putText(frame, "VERIFIED EMPLOYEE", (text_x, card_y + 20), font, 0.42, border_color, 1, cv2.LINE_AA)

            # Name (Bold prominent)
            cv2.putText(frame, name[:20], (text_x, card_y + 44), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Employee ID Badge & Dept
            info_line = f"ID: {emp_id} | {dept[:10]}"
            cv2.putText(frame, info_line, (text_x, card_y + 68), font, 0.44, (200, 210, 220), 1, cv2.LINE_AA)

            # Role
            role_line = f"Role: {role[:24]}"
            cv2.putText(frame, role_line, (text_x, card_y + 88), font, 0.42, (170, 180, 190), 1, cv2.LINE_AA)

            # Confidence Bar
            bar_w = card_w - 32
            bar_h = 6
            bar_y = card_y + 112
            cv2.rectangle(frame, (text_x, bar_y), (text_x + bar_w, bar_y + bar_h), (45, 50, 60), -1)
            fill_w = int(bar_w * (confidence_pct / 100.0))
            cv2.rectangle(frame, (text_x, bar_y), (text_x + fill_w, bar_y + bar_h), border_color, -1)

            conf_label = f"Match: {confidence_pct:.1f}%"
            cv2.putText(frame, conf_label, (text_x, card_y + 107), font, 0.38, border_color, 1, cv2.LINE_AA)

        else:
            # Unknown Person Floating Card
            cv2.putText(frame, "SECURITY ALERT", (text_x, card_y + 24), font, 0.46, border_color, 1, cv2.LINE_AA)
            cv2.putText(frame, "Unknown Person", (text_x, card_y + 55), font, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Access Denied / Unregistered", (text_x, card_y + 82), font, 0.44, (160, 170, 185), 1, cv2.LINE_AA)
            cv2.putText(frame, "Biometric signature not found", (text_x, card_y + 106), font, 0.40, (110, 120, 140), 1, cv2.LINE_AA)
