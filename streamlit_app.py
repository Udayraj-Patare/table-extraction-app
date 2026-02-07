"""
Table Extraction Web App - Production Ready
Upload images or PDFs and extract tables to Excel
Shareable link for team collaboration
"""

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

# PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False

# Import extraction modules
try:
    from table_extraction_pipeline import ImagePreprocessor, TableExtractor
    from advanced_config import AdvancedPreprocessor, QualityAssessment
except ImportError:
    # Fallback if modules aren't available
    st.error("⚠️ Required modules not found. Please ensure all files are uploaded.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Table Extraction Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class TableExtractionApp:
    """Main application class"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.advanced_preprocessor = AdvancedPreprocessor()
        self.quality_assessor = QualityAssessment()
    
    def pdf_to_images(self, pdf_file):
        """Convert PDF to images"""
        if not PDF_SUPPORTED:
            return None, "PDF support not available. Please install: pip install pdf2image"
        
        try:
            # Convert PDF to images
            images = convert_from_bytes(
                pdf_file.read(),
                dpi=300,
                fmt='jpeg'
            )
            return images, None
        except Exception as e:
            return None, f"PDF conversion failed: {str(e)}"
    
    def process_image(self, image_data, filename, settings):
        """Process a single image and extract table"""
        try:
            # Convert to numpy array if PIL Image
            if hasattr(image_data, 'mode'):
                image = np.array(image_data)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Already numpy array
                image = image_data
            
            if image is None:
                return None, "Failed to read image"
            
            # Assess quality
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            quality = self.preprocessor.assess_quality(gray)
            
            # Check if handwritten
            is_handwritten = self.advanced_preprocessor.detect_handwritten_text(gray)
            
            # Preprocess based on settings
            if is_handwritten and settings['enable_handwritten']:
                processed = self.advanced_preprocessor.enhance_handwritten(gray)
            else:
                enhanced = self.preprocessor.enhance_image(image)
                deskewed = self.preprocessor.deskew_image(enhanced)
                processed = self.preprocessor.remove_borders(deskewed)
            
            # Perspective correction
            processed = self.advanced_preprocessor.correct_perspective(processed)
            
            # Extract table using OCR
            extractor = TableExtractor(settings['ocr_engine'], settings['language'])
            df = extractor.extract_table(processed)
            
            if df.empty:
                return None, "No table detected in image"
            
            # Add metadata
            df['source_file'] = filename
            df['page_quality'] = round(quality, 3)
            
            # Clean data
            if settings['enable_validation']:
                df = self.quality_assessor.clean_extracted_data(df)
            
            return {
                'df': df,
                'quality': quality,
                'is_handwritten': is_handwritten,
                'filename': filename
            }, None
            
        except Exception as e:
            return None, f"Processing error: {str(e)}"


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Table Extraction Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 1.2rem;">Upload images or PDFs • Extract tables to Excel • Share with your team</p>',
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # OCR Engine
        ocr_engine = st.selectbox(
            "OCR Engine",
            ["paddle", "tesseract"],
            index=0,
            help="PaddleOCR: More accurate, better for handwritten | Tesseract: Faster"
        )
        
        # Language
        language = st.selectbox(
            "Document Language",
            ["en", "ch", "fr", "de", "es", "ja", "ko"],
            format_func=lambda x: {
                "en": "🇬🇧 English",
                "ch": "🇨🇳 Chinese",
                "fr": "🇫🇷 French",
                "de": "🇩🇪 German",
                "es": "🇪🇸 Spanish",
                "ja": "🇯🇵 Japanese",
                "ko": "🇰🇷 Korean"
            }.get(x, x),
            index=0
        )
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            enable_handwritten = st.checkbox(
                "Handwritten Text Support",
                value=True,
                help="Better processing for handwritten content"
            )
            
            enable_validation = st.checkbox(
                "Quality Validation",
                value=True,
                help="Assess extraction quality"
            )
            
            combine_mode = st.radio(
                "Output Format",
                ["Single sheet (all tables)", "Multiple sheets (one per file)"],
                index=0
            )
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. **Upload** files (max 10)
        2. **Click** Extract Tables
        3. **Download** Excel file
        4. **Share** this link with others!
        """)
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.info("""
        ✅ 300+ DPI scans  
        ✅ Clear images  
        ✅ Straight pages  
        ✅ Good lighting
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose images or PDF files (max 10)",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'pdf'],
            accept_multiple_files=True,
            help="Supported: JPG, PNG, TIFF, BMP, PDF"
        )
        
        if uploaded_files:
            if len(uploaded_files) > 10:
                st.error("⚠️ Maximum 10 files allowed. Only first 10 will be processed.")
                uploaded_files = uploaded_files[:10]
    
    with col2:
        st.markdown("### 📊 Status")
        if uploaded_files:
            st.metric("Files Uploaded", len(uploaded_files))
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.metric("Total Size", f"{total_size:.1f} MB")
            
            # Check for PDFs
            pdf_count = sum(1 for f in uploaded_files if f.name.lower().endswith('.pdf'))
            if pdf_count > 0:
                if PDF_SUPPORTED:
                    st.success(f"✅ {pdf_count} PDF(s) detected")
                else:
                    st.warning("⚠️ PDF support not available. Install pdf2image.")
        else:
            st.info("No files uploaded yet")
    
    # Display uploaded files
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 📋 Files Preview")
        
        # Create preview grid
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files[:8]):
            with cols[idx % 4]:
                if file.type.startswith('image'):
                    st.image(file, caption=file.name, use_container_width=True)
                else:
                    st.info(f"📄 {file.name}\n({file.size / 1024:.1f} KB)")
        
        if len(uploaded_files) > 8:
            st.caption(f"+ {len(uploaded_files) - 8} more files")
    
    # Extract button
    st.markdown("---")
    
    if uploaded_files and st.button("🚀 Extract Tables", type="primary", use_container_width=True):
        
        # Initialize app
        app = TableExtractionApp()
        
        # Collect settings
        settings = {
            'ocr_engine': ocr_engine,
            'language': language,
            'enable_handwritten': enable_handwritten,
            'enable_validation': enable_validation,
            'combine_mode': combine_mode
        }
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        errors = []
        
        # Process each file
        total_files = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.markdown(f"**Processing {idx+1}/{total_files}:** `{uploaded_file.name}`")
            progress_bar.progress((idx + 1) / total_files)
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Handle PDF files
            if uploaded_file.name.lower().endswith('.pdf'):
                images, error = app.pdf_to_images(uploaded_file)
                
                if error:
                    errors.append({'file': uploaded_file.name, 'error': error})
                    continue
                
                # Process each page
                for page_num, image in enumerate(images, 1):
                    page_filename = f"{uploaded_file.name} - Page {page_num}"
                    result, error = app.process_image(image, page_filename, settings)
                    
                    if error:
                        errors.append({'file': page_filename, 'error': error})
                    else:
                        results.append(result)
            
            else:
                # Process image file
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                result, error = app.process_image(image, uploaded_file.name, settings)
                
                if error:
                    errors.append({'file': uploaded_file.name, 'error': error})
                else:
                    results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.processed_results = {
            'results': results,
            'errors': errors,
            'settings': settings
        }
    
    # Display results
    if st.session_state.processed_results:
        results = st.session_state.processed_results['results']
        errors = st.session_state.processed_results['errors']
        settings = st.session_state.processed_results['settings']
        
        st.markdown("---")
        st.markdown("## ✅ Extraction Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", len(results))
        with col2:
            st.metric("❌ Failed", len(errors))
        with col3:
            total_rows = sum(len(r['df']) for r in results) if results else 0
            st.metric("📊 Total Rows", total_rows)
        with col4:
            avg_quality = np.mean([r['quality'] for r in results]) if results else 0
            st.metric("⭐ Avg Quality", f"{avg_quality:.2f}")
        
        if results:
            # Quality breakdown
            if settings['enable_validation']:
                st.markdown("### 📈 Quality Analysis")
                
                quality_data = []
                for r in results:
                    quality_data.append({
                        'File': r['filename'],
                        'Quality': f"{r['quality']:.2f}",
                        'Handwritten': '✓' if r['is_handwritten'] else '✗',
                        'Rows': len(r['df'])
                    })
                
                st.dataframe(quality_data, use_container_width=True)
            
            # Preview data
            st.markdown("### 👀 Data Preview")
            
            all_dfs = [r['df'] for r in results]
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            st.dataframe(combined_df.head(50), use_container_width=True)
            st.caption(f"Showing first 50 of {len(combined_df)} rows")
            
            # Create Excel file
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if settings['combine_mode'].startswith("Single"):
                    # Single sheet
                    combined_df.to_excel(writer, sheet_name='Extracted_Tables', index=False)
                else:
                    # Multiple sheets
                    for idx, r in enumerate(results):
                        sheet_name = f"Table_{idx+1}"[:31]
                        r['df'].to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Quality report
                if settings['enable_validation'] and quality_data:
                    pd.DataFrame(quality_data).to_excel(
                        writer, sheet_name='Quality_Report', index=False
                    )
                
                # Summary
                summary = pd.DataFrame({
                    'Metric': [
                        'Files Processed',
                        'Successful',
                        'Failed',
                        'Total Rows',
                        'Average Quality'
                    ],
                    'Value': [
                        len(results) + len(errors),
                        len(results),
                        len(errors),
                        len(combined_df),
                        f"{avg_quality:.3f}"
                    ]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # Errors
                if errors:
                    pd.DataFrame(errors).to_excel(
                        writer, sheet_name='Errors', index=False
                    )
            
            output.seek(0)
            
            # Download button
            st.markdown("---")
            st.download_button(
                label="📥 Download Excel File",
                data=output.getvalue(),
                file_name=f"extracted_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
            
            st.success("✅ Extraction complete! Click above to download your Excel file.")
        
        # Show errors
        if errors:
            st.markdown("---")
            st.markdown("### ⚠️ Errors")
            st.dataframe(errors, use_container_width=True)
            
            with st.expander("💡 Troubleshooting Tips"):
                st.markdown("""
                **Common fixes:**
                - Re-scan at higher resolution (300+ DPI)
                - Ensure images are clear and straight
                - Check if files are corrupted
                - Try enabling handwritten support
                - Use different OCR engine
                """)
    
    elif not uploaded_files:
        # Welcome screen
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            ### 👋 Welcome!
            
            This tool helps you extract tables from scanned images and PDFs.
            
            **Get started:**
            1. Upload your files above
            2. Adjust settings in sidebar
            3. Click 'Extract Tables'
            4. Download your Excel file
            
            **Share this link** with your team for collaboration!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #999; padding: 1rem;'>"
        "🔒 Your files are processed securely and not stored | "
        "Built with ❤️ using Streamlit & PaddleOCR"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
