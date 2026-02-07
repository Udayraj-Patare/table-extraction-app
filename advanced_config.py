"""
Advanced Table Extraction Configuration
Handles handwritten text, skewed tables, and low-quality scans
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from scipy import ndimage


class AdvancedPreprocessor:
    """Enhanced preprocessing for challenging images"""
    
    @staticmethod
    def detect_handwritten_text(image: np.ndarray) -> bool:
        """Detect if image likely contains handwritten text"""
        # Calculate edge density and irregularity
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Handwritten text typically has higher edge density
        return edge_density > 0.15
    
    @staticmethod
    def enhance_handwritten(image: np.ndarray) -> np.ndarray:
        """Special preprocessing for handwritten text"""
        # Bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Morphological operations to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # Adaptive threshold with larger block size
        binary = cv2.adaptiveThreshold(
            morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        
        return binary
    
    @staticmethod
    def correct_perspective(image: np.ndarray) -> np.ndarray:
        """Correct perspective distortion in scanned documents"""
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100,
            minLineLength=100, maxLineGap=10
        )
        
        if lines is None:
            return image
        
        # Calculate dominant angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Get median angle
        if angles:
            median_angle = np.median(angles)
            
            # Rotate if significant skew
            if abs(median_angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image
    
    @staticmethod
    def enhance_low_quality(image: np.ndarray) -> np.ndarray:
        """Enhance very poor quality scans"""
        # Super resolution using interpolation
        scale_factor = 2
        height, width = image.shape[:2]
        upscaled = cv2.resize(
            image, 
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        # Reduce noise
        denoised = cv2.fastNlMeansDenoising(sharpened, h=10)
        
        return denoised
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """Remove shadows and uneven illumination"""
        # Dilate to create background model
        dilated = cv2.dilate(image, np.ones((7,7), np.uint8))
        
        # Median blur for smooth background
        bg = cv2.medianBlur(dilated, 21)
        
        # Subtract background
        diff = 255 - cv2.absdiff(image, bg)
        
        # Normalize
        normalized = cv2.normalize(
            diff, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        return normalized


class SmartTableDetector:
    """Intelligent table structure detection"""
    
    @staticmethod
    def detect_table_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table regions in image"""
        # Threshold
        _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours of table regions
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and return bounding boxes
        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum table size
                x, y, w, h = cv2.boundingRect(contour)
                tables.append((x, y, w, h))
        
        return tables
    
    @staticmethod
    def detect_grid_structure(image: np.ndarray) -> Tuple[List[int], List[int]]:
        """Detect row and column separators"""
        # Project pixels horizontally and vertically
        horizontal_projection = np.sum(image == 0, axis=1)
        vertical_projection = np.sum(image == 0, axis=0)
        
        # Find peaks (separators)
        h_threshold = np.mean(horizontal_projection) + np.std(horizontal_projection)
        v_threshold = np.mean(vertical_projection) + np.std(vertical_projection)
        
        row_separators = np.where(horizontal_projection > h_threshold)[0]
        col_separators = np.where(vertical_projection > v_threshold)[0]
        
        # Group consecutive separators
        def group_separators(separators, min_gap=10):
            if len(separators) == 0:
                return []
            
            groups = []
            current_group = [separators[0]]
            
            for i in range(1, len(separators)):
                if separators[i] - current_group[-1] < min_gap:
                    current_group.append(separators[i])
                else:
                    groups.append(int(np.mean(current_group)))
                    current_group = [separators[i]]
            
            groups.append(int(np.mean(current_group)))
            return groups
        
        rows = group_separators(row_separators)
        cols = group_separators(col_separators)
        
        return rows, cols
    
    @staticmethod
    def extract_cells(image: np.ndarray, rows: List[int], cols: List[int]) -> List[List[np.ndarray]]:
        """Extract individual cells from grid"""
        cells = []
        
        # Add image boundaries
        row_bounds = [0] + rows + [image.shape[0]]
        col_bounds = [0] + cols + [image.shape[1]]
        
        for i in range(len(row_bounds) - 1):
            row_cells = []
            for j in range(len(col_bounds) - 1):
                cell = image[
                    row_bounds[i]:row_bounds[i+1],
                    col_bounds[j]:col_bounds[j+1]
                ]
                row_cells.append(cell)
            cells.append(row_cells)
        
        return cells


class QualityAssessment:
    """Assess and improve extraction quality"""
    
    @staticmethod
    def validate_table_structure(df: pd.DataFrame) -> Dict[str, float]:
        """Validate extracted table quality"""
        metrics = {}
        
        # Completeness: ratio of non-empty cells
        total_cells = df.size
        non_empty = df.astype(str).apply(lambda x: x.str.strip() != '').sum().sum()
        metrics['completeness'] = non_empty / total_cells if total_cells > 0 else 0
        
        # Consistency: coefficient of variation in row lengths
        row_lengths = df.apply(lambda x: x.astype(str).str.len().sum(), axis=1)
        metrics['consistency'] = 1 - (row_lengths.std() / row_lengths.mean() 
                                     if row_lengths.mean() > 0 else 1)
        
        # Uniformity: how uniform column widths are
        col_widths = df.apply(lambda x: x.astype(str).str.len().mean())
        metrics['uniformity'] = 1 - (col_widths.std() / col_widths.mean() 
                                    if col_widths.mean() > 0 else 1)
        
        # Overall quality score
        metrics['overall'] = np.mean([
            metrics['completeness'],
            metrics['consistency'],
            metrics['uniformity']
        ])
        
        return metrics
    
    @staticmethod
    def clean_extracted_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize extracted data"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Try to infer data types
        for col in df.columns:
            # Try numeric conversion
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df
    
    @staticmethod
    def suggest_improvements(quality_metrics: Dict[str, float]) -> List[str]:
        """Suggest improvements based on quality metrics"""
        suggestions = []
        
        if quality_metrics['completeness'] < 0.7:
            suggestions.append(
                "Low completeness detected. Try:\n"
                "  - Increase image resolution\n"
                "  - Improve lighting/contrast\n"
                "  - Use bilateral filtering"
            )
        
        if quality_metrics['consistency'] < 0.6:
            suggestions.append(
                "Inconsistent structure detected. Try:\n"
                "  - Check for perspective distortion\n"
                "  - Verify table boundaries\n"
                "  - Apply deskewing"
            )
        
        if quality_metrics['uniformity'] < 0.5:
            suggestions.append(
                "Non-uniform data detected. Try:\n"
                "  - Verify column detection\n"
                "  - Check for merged cells\n"
                "  - Manual verification recommended"
            )
        
        if quality_metrics['overall'] < 0.5:
            suggestions.append(
                "Overall poor quality. Consider:\n"
                "  - Re-scanning with higher DPI (300+ recommended)\n"
                "  - Manual data entry for critical tables\n"
                "  - Using specialized table extraction tools"
            )
        
        return suggestions


class OptimizationConfig:
    """Performance optimization settings"""
    
    # Image preprocessing
    RESIZE_FOR_OCR = True  # Resize large images
    TARGET_WIDTH = 2000  # Max width for OCR
    
    # OCR settings
    CONFIDENCE_THRESHOLD = 0.5  # Min confidence for text
    
    # Batch processing
    BATCH_SIZE = 100  # Images per batch
    PREFETCH_IMAGES = 5  # Number of images to preload
    
    # Memory management
    CLEAR_MEMORY_AFTER = 50  # Clear cache every N images
    
    # Quality thresholds
    MIN_QUALITY_SCORE = 0.3  # Skip very poor images
    RETRY_FAILED = True  # Retry failed images with different settings
    
    # Output settings
    SAVE_INTERMEDIATE = False  # Save preprocessed images
    INTERMEDIATE_DIR = "preprocessed"
    
    # Logging
    VERBOSE = True
    LOG_EVERY_N = 10  # Log progress every N images


# Export configurations
__all__ = [
    'AdvancedPreprocessor',
    'SmartTableDetector',
    'QualityAssessment',
    'OptimizationConfig'
]
