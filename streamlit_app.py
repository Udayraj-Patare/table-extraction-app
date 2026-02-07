"""
Table Extraction Web App - Production Ready
Upload images or PDFs and extract tables to Excel
Shareable link for team collaboration
"""

# =============================
# SAFE PATH SETUP (CRITICAL)
# =============================
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================
# LIGHTWEIGHT IMPORTS ONLY
# =============================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime
import base64
from io import BytesIO
import time

# =============================
# LAZY LOAD OCR DEPENDENCIES
# =============================
@st.cache_resource
def load_ocr_dependencies():
    from paddleocr import PaddleOCR
    import pytesseract
    from pdf2image import convert_from_bytes
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return ocr, pytesseract, convert_from_bytes

# =============================
# PDF SUPPORT CHECK
# =============================
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORTED = True
except Exception:
    PDF_SUPPORTED = False

# =============================
# IMPORT EXTRACTION MODULES
# =============================
try:
    from table_extraction_pipeline import ImagePreprocessor, TableExtractor
    from advanced_config import AdvancedPreprocessor, QualityAssessment
except ImportError as e:
    st.error(f"⚠️ Required modules not found: {e}")
    st.stop()

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Table Extraction Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =============================
# MAIN APP CLASS
# =============================
class TableExtractionApp:

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.advanced_preprocessor = AdvancedPreprocessor()
        self.quality_assessor = QualityAssessment()

    def pdf_to_images(self, pdf_file):
        if not PDF_SUPPORTED:
            return None, "PDF support not available"

        try:
            _, _, convert_from_bytes = load_ocr_dependencies()
            images = convert_from_bytes(pdf_file.read(), dpi=300)
            return images, None
        except Exception as e:
            return None, str(e)

    def process_image(self, image_data, filename, settings):
        try:
            # Convert image
            if hasattr(image_data, "mode"):
                image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            else:
                image = image_data

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            quality = self.preprocessor.assess_quality(gray)
            is_handwritten = self.advanced_preprocessor.detect_handwritten_text(gray)

            if is_handwritten and settings["enable_handwritten"]:
                processed = self.advanced_preprocessor.enhance_handwritten(gray)
            else:
                processed = self.preprocessor.remove_borders(
                    self.preprocessor.deskew_image(
                        self.preprocessor.enhance_image(image)
                    )
                )

            processed = self.advanced_preprocessor.correct_perspective(processed)

            extractor = TableExtractor(settings["ocr_engine"], settings["language"])
            df = extractor.extract_table(processed)

            if df.empty:
                return None, "No table detected"

            df["source_file"] = filename
            df["page_quality"] = round(quality, 3)

            if settings["enable_validation"]:
                df = self.quality_assessor.clean_extracted_data(df)

            return {
                "df": df,
                "quality": quality,
                "is_handwritten": is_handwritten,
                "filename": filename
            }, None

        except Exception as e:
            return None, str(e)

# =============================
# MAIN FUNCTION
# =============================
def main():
    st.markdown('<div class="main-header">📊 Table Extraction Tool</div>', unsafe_allow_html=True)

    if "processed_results" not in st.session_state:
        st.session_state.processed_results = None

    with st.sidebar:
        st.header("⚙️ Settings")
        ocr_engine = st.selectbox("OCR Engine", ["paddle", "tesseract"])
        language = st.selectbox("Language", ["en", "fr", "de", "es"])
        enable_handwritten = st.checkbox("Handwritten Support", True)
        enable_validation = st.checkbox("Quality Validation", True)
        combine_mode = st.radio("Output", ["Single sheet", "Multiple sheets"])

    uploaded_files = st.file_uploader(
        "Upload Images or PDFs",
        type=["jpg", "png", "jpeg", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Extract Tables"):
        app = TableExtractionApp()
        results, errors = [], []

        settings = {
            "ocr_engine": ocr_engine,
            "language": language,
            "enable_handwritten": enable_handwritten,
            "enable_validation": enable_validation,
            "combine_mode": combine_mode
        }

        for file in uploaded_files:
            file.seek(0)

            if file.name.lower().endswith(".pdf"):
                images, err = app.pdf_to_images(file)
                if err:
                    errors.append({"file": file.name, "error": err})
                    continue
                for i, img in enumerate(images):
                    res, err = app.process_image(img, f"{file.name}-p{i+1}", settings)
                    if err:
                        errors.append({"file": file.name, "error": err})
                    else:
                        results.append(res)
            else:
                img = cv2.imdecode(
                    np.frombuffer(file.read(), np.uint8),
                    cv2.IMREAD_COLOR
                )
                res, err = app.process_image(img, file.name, settings)
                if err:
                    errors.append({"file": file.name, "error": err})
                else:
                    results.append(res)

        st.session_state.processed_results = {"results": results, "errors": errors}

    if st.session_state.processed_results:
        results = st.session_state.processed_results["results"]
        errors = st.session_state.processed_results["errors"]

        if results:
            combined = pd.concat([r["df"] for r in results], ignore_index=True)
            st.dataframe(combined.head(50), use_container_width=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                combined.to_excel(writer, index=False, sheet_name="Extracted_Tables")

            st.download_button(
                "📥 Download Excel",
                data=output.getvalue(),
                file_name=f"tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if errors:
            st.warning("Some files failed")
            st.dataframe(errors)

if __name__ == "__main__":
    main()
