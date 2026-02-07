"""
Production-Ready Table Extraction Pipeline
Extracts tabular data from scanned images and exports to Excel
Supports batch processing of thousands of images with error handling
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import traceback
from dataclasses import dataclass
import json

# OCR engines
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not available. Install with: pip install paddleocr")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: Tesseract not available. Install with: pip install pytesseract pillow")


@dataclass
class ProcessingResult:
    """Result of processing a single image"""
    image_path: str
    success: bool
    data: Optional[pd.DataFrame]
    error: Optional[str]
    processing_time: float
    quality_score: float


class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR accuracy"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Apply comprehensive preprocessing pipeline"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Binarization using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Correct skewed/rotated images"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only deskew if angle is significant
        if abs(angle) < 0.5:
            return image
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    @staticmethod
    def remove_borders(image: np.ndarray) -> np.ndarray:
        """Remove black borders and artifacts"""
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # Find largest contour (assumed to be the document)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop to content
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    @staticmethod
    def assess_quality(image: np.ndarray) -> float:
        """Assess image quality (0-1 scale)"""
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Normalize to 0-1 (empirical threshold: good images > 100)
        quality = min(laplacian_var / 500.0, 1.0)
        
        return quality
    
    def preprocess_pipeline(self, image_path: str) -> Tuple[np.ndarray, float]:
        """Complete preprocessing pipeline"""
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Assess original quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quality = self.assess_quality(gray)
        
        # Apply enhancements
        enhanced = self.enhance_image(image)
        deskewed = self.deskew_image(enhanced)
        cleaned = self.remove_borders(deskewed)
        
        return cleaned, quality


class TableExtractor:
    """Extract tables using OCR and structure detection"""
    
    def __init__(self, ocr_engine: str = "paddle", lang: str = "en"):
        """
        Initialize OCR engine
        
        Args:
            ocr_engine: 'paddle' or 'tesseract'
            lang: Language code for OCR
        """
        self.ocr_engine = ocr_engine
        self.lang = lang
        
        if ocr_engine == "paddle" and PADDLE_AVAILABLE:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=False,  # Set to True if GPU available
                show_log=False
            )
        elif ocr_engine == "tesseract" and TESSERACT_AVAILABLE:
            self.ocr = None  # Tesseract uses function calls
        else:
            raise ValueError(f"OCR engine '{ocr_engine}' not available")
    
    def extract_text_paddle(self, image: np.ndarray) -> List[Dict]:
        """Extract text using PaddleOCR"""
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return []
        
        extracted = []
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Calculate position
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            extracted.append({
                'text': text,
                'confidence': confidence,
                'x_min': min(x_coords),
                'y_min': min(y_coords),
                'x_max': max(x_coords),
                'y_max': max(y_coords),
                'x_center': sum(x_coords) / 4,
                'y_center': sum(y_coords) / 4
            })
        
        return extracted
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract"""
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Get detailed data
        data = pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT,
            lang=self.lang
        )
        
        extracted = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Filter low confidence
                text = data['text'][i].strip()
                if text:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    extracted.append({
                        'text': text,
                        'confidence': float(data['conf'][i]) / 100.0,
                        'x_min': x,
                        'y_min': y,
                        'x_max': x + w,
                        'y_max': y + h,
                        'x_center': x + w / 2,
                        'y_center': y + h / 2
                    })
        
        return extracted
    
    def detect_table_structure(self, text_boxes: List[Dict]) -> pd.DataFrame:
        """Convert text boxes to structured table"""
        if not text_boxes:
            return pd.DataFrame()
        
        # Sort by Y position (rows), then X position (columns)
        sorted_boxes = sorted(text_boxes, key=lambda x: (x['y_center'], x['x_center']))
        
        # Group into rows based on Y proximity
        rows = []
        current_row = []
        y_threshold = 20  # Pixels
        
        for i, box in enumerate(sorted_boxes):
            if i == 0:
                current_row.append(box)
            else:
                y_diff = abs(box['y_center'] - sorted_boxes[i-1]['y_center'])
                
                if y_diff < y_threshold:
                    current_row.append(box)
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [box]
        
        if current_row:
            rows.append(current_row)
        
        # Sort cells within each row by X position
        for row in rows:
            row.sort(key=lambda x: x['x_center'])
        
        # Determine maximum columns
        max_cols = max(len(row) for row in rows) if rows else 0
        
        # Create DataFrame
        table_data = []
        for row in rows:
            row_data = [cell['text'] for cell in row]
            # Pad with empty strings if needed
            row_data.extend([''] * (max_cols - len(row_data)))
            table_data.append(row_data)
        
        if not table_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(table_data)
        
        # Try to detect header row (usually first row or row with most text)
        if len(df) > 1:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        return df
    
    def extract_table(self, image: np.ndarray) -> pd.DataFrame:
        """Main extraction method"""
        if self.ocr_engine == "paddle":
            text_boxes = self.extract_text_paddle(image)
        else:
            text_boxes = self.extract_text_tesseract(image)
        
        return self.detect_table_structure(text_boxes)


class TableExtractionPipeline:
    """Complete pipeline for batch processing"""
    
    def __init__(
        self,
        input_dir: str,
        output_file: str,
        ocr_engine: str = "paddle",
        lang: str = "en",
        max_workers: int = 4,
        log_dir: str = "logs"
    ):
        """
        Initialize pipeline
        
        Args:
            input_dir: Directory containing images
            output_file: Output Excel file path
            ocr_engine: 'paddle' or 'tesseract'
            lang: Language code
            max_workers: Number of parallel workers
            log_dir: Directory for logs
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.ocr_engine = ocr_engine
        self.lang = lang
        self.max_workers = max_workers
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.extractor = TableExtractor(ocr_engine, lang)
    
    def _setup_logging(self):
        """Configure logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_image_files(self) -> List[Path]:
        """Get all image files from input directory"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in self.input_dir.rglob('*')
            if f.suffix.lower() in extensions
        ]
        return sorted(image_files)
    
    def process_single_image(self, image_path: Path) -> ProcessingResult:
        """Process a single image"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing: {image_path.name}")
            
            # Preprocess
            processed_image, quality = self.preprocessor.preprocess_pipeline(str(image_path))
            
            # Extract table
            df = self.extractor.extract_table(processed_image)
            
            # Check if extraction was successful
            if df.empty:
                raise ValueError("No table data extracted")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add metadata columns
            df['source_image'] = image_path.name
            df['quality_score'] = quality
            
            return ProcessingResult(
                image_path=str(image_path),
                success=True,
                data=df,
                error=None,
                processing_time=processing_time,
                quality_score=quality
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Failed to process {image_path.name}: {error_msg}")
            
            return ProcessingResult(
                image_path=str(image_path),
                success=False,
                data=None,
                error=error_msg,
                processing_time=processing_time,
                quality_score=0.0
            )
    
    def process_batch(self, image_files: List[Path]) -> List[ProcessingResult]:
        """Process images in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_single_image, img)
                for img in image_files
            ]
            
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5 min timeout per image
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
        
        return results
    
    def save_results(self, results: List[ProcessingResult]):
        """Save all results to Excel"""
        # Combine all successful DataFrames
        successful_dfs = [r.data for r in results if r.success and r.data is not None]
        
        if not successful_dfs:
            self.logger.error("No data to save - all extractions failed")
            return
        
        # Combine all data
        combined_df = pd.concat(successful_dfs, ignore_index=True)
        
        # Save to Excel
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            # Main data sheet
            combined_df.to_excel(writer, sheet_name='Extracted_Data', index=False)
            
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Images',
                    'Successful Extractions',
                    'Failed Extractions',
                    'Total Rows Extracted',
                    'Average Quality Score',
                    'Average Processing Time (s)'
                ],
                'Value': [
                    len(results),
                    sum(1 for r in results if r.success),
                    sum(1 for r in results if not r.success),
                    len(combined_df),
                    f"{np.mean([r.quality_score for r in results if r.success]):.3f}",
                    f"{np.mean([r.processing_time for r in results]):.2f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Error log sheet
            error_data = [
                {
                    'Image': Path(r.image_path).name,
                    'Error': r.error,
                    'Processing Time': f"{r.processing_time:.2f}s"
                }
                for r in results if not r.success
            ]
            if error_data:
                error_df = pd.DataFrame(error_data)
                error_df.to_excel(writer, sheet_name='Errors', index=False)
        
        self.logger.info(f"Results saved to: {self.output_file}")
        self.logger.info(f"Total rows extracted: {len(combined_df)}")
    
    def run(self):
        """Execute the complete pipeline"""
        self.logger.info("="*60)
        self.logger.info("Table Extraction Pipeline Started")
        self.logger.info("="*60)
        
        # Get image files
        image_files = self.get_image_files()
        self.logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            self.logger.error("No images found in input directory")
            return
        
        # Process in batches for memory efficiency
        batch_size = 100
        all_results = []
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} images)")
            
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
        
        # Save results
        self.save_results(all_results)
        
        # Final summary
        self.logger.info("="*60)
        self.logger.info("Pipeline Completed")
        self.logger.info(f"Success rate: {sum(1 for r in all_results if r.success)}/{len(all_results)}")
        self.logger.info("="*60)


# Main execution
if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "scanned_images"  # Directory with your images
    OUTPUT_FILE = "extracted_tables.xlsx"
    OCR_ENGINE = "paddle"  # or "tesseract"
    LANGUAGE = "en"  # Language code
    MAX_WORKERS = 4  # Adjust based on CPU cores
    
    # Run pipeline
    pipeline = TableExtractionPipeline(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        ocr_engine=OCR_ENGINE,
        lang=LANGUAGE,
        max_workers=MAX_WORKERS
    )
    
    pipeline.run()
