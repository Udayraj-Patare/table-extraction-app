import cv2
import numpy as np
import pandas as pd
import pytesseract
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class ProcessingResult:
    image_path: str
    success: bool
    data: pd.DataFrame | None
    error: str | None
    processing_time: float
    quality_score: float


# =========================
# IMAGE PREPROCESSOR
# =========================
class ImagePreprocessor:

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return image[y:y+h, x:x+w]

    def assess_quality(self, image: np.ndarray) -> float:
        lap = cv2.Laplacian(image, cv2.CV_64F)
        return min(np.var(lap) / 1000.0, 1.0)


# =========================
# TABLE EXTRACTOR
# =========================
class TableExtractor:

    def __init__(self, ocr_engine: str = "paddle", language: str = "en"):
        self.ocr_engine = ocr_engine
        self.language = language

        if ocr_engine == "paddle" and PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=language)
        else:
            self.ocr = None

    # ---------- OCR ----------
    def extract_text_paddle(self, image: np.ndarray) -> List[Dict]:
        if not self.ocr:
            return []

        result = self.ocr.ocr(image, cls=True)
        if not result or not isinstance(result, list):
            return []

        page = result[0] if result else []
        extracted = []

        for line in page:
            if len(line) < 2:
                continue
            bbox = line[0]
            text = line[1][0] if line[1] else ""
            confidence = float(line[1][1]) if len(line[1]) > 1 else 0.0

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            extracted.append({
                "text": text,
                "confidence": confidence,
                "x_center": sum(xs) / len(xs),
                "y_center": sum(ys) / len(ys)
            })

        return extracted

    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        extracted = []

        n = len(data["text"])
        for i in range(n):
            try:
                conf = float(data["conf"][i])
            except Exception:
                continue
            if conf <= 0:
                continue

            text = (data["text"][i] or "").strip()
            if not text:
                continue

            x, y, w, h = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i]
            )

            extracted.append({
                "text": text,
                "confidence": conf / 100.0,
                "x_center": x + w / 2,
                "y_center": y + h / 2
            })

        return extracted

    # ---------- TABLE ----------
    def extract_table(self, image: np.ndarray) -> pd.DataFrame:
        if self.ocr_engine == "paddle" and PADDLE_AVAILABLE:
            words = self.extract_text_paddle(image)
        else:
            words = self.extract_text_tesseract(image)

        if not words:
            return pd.DataFrame()

        words = sorted(words, key=lambda w: (w["y_center"], w["x_center"]))
        rows, current_row = [], []
        y_threshold = 15

        for w in words:
            if not current_row:
                current_row.append(w)
                continue

            if abs(w["y_center"] - current_row[-1]["y_center"]) < y_threshold:
                current_row.append(w)
            else:
                rows.append(current_row)
                current_row = [w]

        if current_row:
            rows.append(current_row)

        table = []
        for row in rows:
            row_sorted = sorted(row, key=lambda x: x["x_center"])
            table.append([w["text"] for w in row_sorted])

        df = pd.DataFrame(table)

        # header handling (safe)
        if len(df) > 1:
            header = df.iloc[0].astype(str).tolist()
            seen, clean = {}, []
            for h in header:
                base = h.strip() or "column"
                name = base
                i = 1
                while name in seen:
                    name = f"{base}_{i}"
                    i += 1
                seen[name] = True
                clean.append(name)
            df.columns = clean
            df = df.iloc[1:].reset_index(drop=True)

        return df


# =========================
# PIPELINE
# =========================
class TableExtractionPipeline:

    def __init__(self, input_dir: Path, output_dir: Path, max_workers: int = 2):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers

        self.log_dir = self.output_dir / "logs"
        self.output_file = self.output_dir / "extracted_tables.xlsx"

        # ✅ FIXED DIRECTORY CREATION
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.preprocessor = ImagePreprocessor()
        self.extractor = TableExtractor()

    # ---------- logging ----------
    def _setup_logging(self):
        self.logger = logging.getLogger("table_pipeline")
        if self.logger.handlers:
            return

        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        fh = logging.FileHandler(self.log_dir / "pipeline.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    # ---------- processing ----------
    def process_single_image(self, image_path: Path) -> ProcessingResult:
        start = datetime.now()

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Image could not be read")

            enhanced = self.preprocessor.enhance_image(image)
            deskewed = self.preprocessor.deskew_image(enhanced)
            cleaned = self.preprocessor.remove_borders(deskewed)

            df = self.extractor.extract_table(cleaned)
            quality = self.preprocessor.assess_quality(cleaned)

            return ProcessingResult(
                image_path=str(image_path),
                success=not df.empty,
                data=df if not df.empty else None,
                error=None if not df.empty else "No table detected",
                processing_time=(datetime.now() - start).total_seconds(),
                quality_score=quality
            )

        except Exception as e:
            return ProcessingResult(
                image_path=str(image_path),
                success=False,
                data=None,
                error=str(e),
                processing_time=(datetime.now() - start).total_seconds(),
                quality_score=0.0
            )

    def process_batch(self, images: List[Path]) -> List[ProcessingResult]:
        results = []

        # ✅ ThreadPoolExecutor avoids pickle crashes
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_single_image, img): img for img in images}

            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    img = futures[f]
                    self.logger.error(f"Failed {img}: {e}")

        return results
