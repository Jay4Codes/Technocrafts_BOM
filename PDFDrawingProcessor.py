import fitz
import numpy as np
import cv2
from paddleocr import PaddleOCR
import re
import pandas as pd
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class PDFDrawingProcessor:
    def __init__(self, pdf_path=None, pdf_data=None, rois=None):
        self.rois = rois or {
            "rev": [(0.427, 0.451, 0.9, 0.955)],
            "title": [(0.745, 0.959, 0.872, 0.937)],
            "sheet": [(0.887, 0.959, 0.930, 0.952)],
            "dnum": [(0.678, 0.75, 0.92, 0.955)],
        }
        
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully")
            print("[INFO] PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
            raise
            
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        self.temp_file_path = None
        
        if pdf_path:
            self.pdf_path = pdf_path
            try:
                self.doc = fitz.open(pdf_path)
                logger.info(f"PDF opened successfully: {pdf_path}")
                print(f"[INFO] PDF opened successfully: {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to open PDF: {e}")
                print(f"[ERROR] Failed to open PDF: {e}")
                raise
        elif pdf_data:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(pdf_data)
                    self.temp_file_path = temp_file.name
                    self.pdf_path = temp_file.name
                    self.doc = fitz.open(self.pdf_path)
                logger.info("PDF created from data successfully")
                print("[INFO] PDF created from data successfully")
            except Exception as e:
                logger.error(f"Failed to create PDF from data: {e}")
                print(f"[ERROR] Failed to create PDF from data: {e}")
                raise
        else:
            raise ValueError("Either pdf_path or pdf_data must be provided")
            
        self.results = {}
        self.rev_per_page = []
        self.temp_title_texts = {}
        self.temp_sheet_num = {}
        self.page_texts_dnum = {}
        self.index_dict = {}
        self.rev_details = []
        self.title_details = []
        self.sheet_details = []
        self.dnum_details = []

    def __del__(self):
        try:
            if hasattr(self, 'doc') and self.doc:
                self.doc.close()
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)
                logger.info("Temporary file cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def _load_pdf_page_as_image(self, page_number, dpi=300):
        page = self.doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError("Unsupported color format")

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def _extract_and_dilate_roi(self, thresh, roi_ratio):
        height, width = thresh.shape
        sx, ex, sy, ey = roi_ratio
        start_x = int(width * sx)
        end_x = int(width * ex)
        start_y = int(height * sy)
        end_y = int(height * ey)
        roi = thresh[start_y:end_y, start_x:end_x]
        dilated_roi = cv2.dilate(roi, self.kernel, iterations=1)
        return dilated_roi, (start_x, end_x, start_y, end_y)

    def _visualize_and_crop_contours(self, image, dilated_roi, roi_coords):
        start_x, end_x, start_y, end_y = roi_coords
        contours, _ = cv2.findContours(dilated_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_vis = image.copy()
        for cnt in contours:
            cnt += [start_x, start_y]
            cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 0), 2)
        zoomed_roi = contour_vis[start_y:end_y, start_x:end_x]
        return zoomed_roi

    def _preprocess_for_ocr(self, zoomed_roi):
        gray = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.5)
        sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)
        thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def _run_ocr(self, image):
        result = self.ocr.predict(image)
        if not result or not result[0]:
            return [], []
        
        print("rec_texts", result[0]['rec_texts'])
        print("rec_scores", result[0]['rec_scores'])
        print("rec_polys", result[0]['rec_polys'])
        return result[0]['rec_texts'], result[0]['rec_scores'], result[0]['rec_polys']

    def run_all_rois(self, pages=None):
        try:
            if pages is None:
                pages = list(range(len(self.doc)))
            
            logger.info(f"Processing {len(pages)} pages for ROI extraction")
            print(f"[INFO] Processing {len(pages)} pages for ROI extraction")

            for roi_key, roi_list in self.rois.items():
                logger.info(f"Processing ROI: {roi_key}")
                print(f"[INFO] Processing ROI: {roi_key}")

                for page_number in pages:
                    try:
                        logger.info(f"Processing page {page_number + 1}/{len(self.doc)} for {roi_key}")
                        print(f"[INFO] Processing page {page_number + 1}/{len(self.doc)} for {roi_key}")
                        
                        image = self._load_pdf_page_as_image(page_number)
                        if image is None:
                            continue
                            
                        thresh = self._preprocess_image(image)
                        if thresh is None:
                            continue

                        for roi_ratio in roi_list:
                            dilated_roi, roi_coords = self._extract_and_dilate_roi(thresh, roi_ratio)
                            zoomed = self._visualize_and_crop_contours(image, dilated_roi, roi_coords)
                            preprocessed = self._preprocess_for_ocr(zoomed)
                            texts, scores, polys = self._run_ocr(preprocessed)

                            if roi_key == 'rev':
                                rev = [int(x) for x in texts if x.isdigit() or x == '0']
                                self.rev_per_page.append(max(rev) if rev else 0)
                                self.rev_details.append((page_number, texts, scores, polys))
                            elif roi_key == 'title':
                                texts = texts[1:] if texts and texts[0] in ['TITLE', 'TTTLE'] else texts
                                self.temp_title_texts[page_number] = ' '.join([t.strip() for t in texts if t.strip()])
                                self.title_details.append((page_number, texts, scores, polys))
                            elif roi_key == 'sheet':
                                cleaned = []
                                for text in texts:
                                    text = text.replace('SHEET', '').strip()
                                    text = re.sub(r'\bNO\.?\b', '', text).strip()
                                    if text and text != '.':
                                        cleaned.append(text)
                                page_number_pattern = re.compile(r'^\d{1,3}[A-Z]?$')
                                for item in cleaned:
                                    if page_number_pattern.fullmatch(item):
                                        self.temp_sheet_num[page_number] = item
                                        break
                                self.sheet_details.append((page_number, texts, scores, polys))
                            elif roi_key == 'dnum':
                                cleaned = ' '.join(t.strip() for t in texts if t.strip())
                                cleaned = cleaned.replace("DRAWING NO.", "").replace("DRAWINGNO.", "").strip()
                                self.page_texts_dnum[page_number] = cleaned
                                self.dnum_details.append((page_number, texts, scores, polys))
                                
                    except Exception as e:
                        logger.error(f"Error processing page {page_number} for {roi_key}: {e}")
                        print(f"[ERROR] Error processing page {page_number} for {roi_key}: {e}")
                        continue
            
            self._build_index_dict()
            logger.info("ROI processing completed successfully")
            print("[INFO] ROI processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in run_all_rois: {e}")
            print(f"[ERROR] Error in run_all_rois: {e}")
            raise

    def _build_index_dict(self):
        try:
            code_map = {}
            sequence_counter = 0
            for k, v in self.page_texts_dnum.items():
                normalized = v.replace(" ", "") if v else ""
                if normalized not in code_map:
                    code_map[normalized] = sequence_counter
                    sequence_counter += 1
                self.index_dict[k] = code_map[normalized]
            logger.info(f"Built index dictionary with {len(self.index_dict)} entries")
            print(f"[INFO] Built index dictionary with {len(self.index_dict)} entries")
        except Exception as e:
            logger.error(f"Error building index dictionary: {e}")
            print(f"[ERROR] Error building index dictionary: {e}")

    def get_dataframe(self):
        try:
            max_pages = len(self.doc)
            
            data = []
            for page_num in range(max_pages):
                row = {
                    'SR NO.': self.index_dict.get(page_num, ''),
                    'DRAWING NO.': self.page_texts_dnum.get(page_num, ''),
                    'SHEET': self.temp_sheet_num.get(page_num, ''),
                    'REV.': self.rev_per_page[page_num] if page_num < len(self.rev_per_page) else 0,
                    'DRAWING TITLE': self.temp_title_texts.get(page_num, '')
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            logger.info(f"Created dataframe with {len(df)} rows")
            print(f"[INFO] Created dataframe with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error creating dataframe: {e}")
            print(f"[ERROR] Error creating dataframe: {e}")
            return pd.DataFrame()

    def _create_structured_pdf_from_dataframe(self, df):
        try:
            if df.empty:
                logger.warning("Empty dataframe provided for PDF creation")
                print("[WARNING] Empty dataframe provided for PDF creation")
                return None, None

            buffer = io.BytesIO()
            
            try:
                custom_pagesize = self.doc[0].rect
                page_width = custom_pagesize.width
                page_height = custom_pagesize.height
            except Exception as e:
                logger.warning(f"Could not get page size from PDF, using default: {e}")
                print(f"[WARNING] Could not get page size from PDF, using default: {e}")
                page_width, page_height = 612, 792

            doc = SimpleDocTemplate(
                buffer,
                pagesize=(page_width, page_height),
                leftMargin=30,
                rightMargin=30,
                topMargin=30,
                bottomMargin=30,
            )

            para_style = ParagraphStyle(
                name='TableCell',
                fontName='Courier-Bold',
                fontSize=12,
                alignment=1,
                leading=14,
            )

            df = df.astype(str).fillna('')
            data = [df.columns.tolist()] + [
                [Paragraph(str(cell), para_style) for cell in row]
                for row in df.values.tolist()
            ]

            usable_width = 0.75 * (page_width - doc.leftMargin - doc.rightMargin)
            column_width_ratios = [0.10, 0.20, 0.10, 0.10, 0.50]
            col_widths = [usable_width * pct for pct in column_width_ratios]

            table = Table(data, colWidths=col_widths, repeatRows=1)
            table.hAlign = 'CENTER'

            style = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("FONTNAME", (0, 1), (-1, -1), "Courier-Bold"),
                ("FONTSIZE", (0, 1), (-1, -1), 6),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 1), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
                ("TOPPADDING", (0, 1), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ])

            table.setStyle(style)
            doc.build([table])
            buffer.seek(0)
            
            logger.info("PDF table created successfully")
            print("[INFO] PDF table created successfully")
            return fitz.open("pdf", buffer.read()), buffer
            
        except Exception as e:
            logger.error(f"Error creating structured PDF: {e}")
            print(f"[ERROR] Error creating structured PDF: {e}")
            return None, None

    def insert_dataframe_table_to_pdf(self, output_pdf_path):
        try:
            df = self.get_dataframe()
            if df.empty:
                logger.error("Cannot create PDF with empty dataframe")
                print("[ERROR] Cannot create PDF with empty dataframe")
                return None
                
            original_pdf = fitz.open(self.pdf_path)
            table_pdf, buffer = self._create_structured_pdf_from_dataframe(df)
            
            if table_pdf is None or buffer is None:
                logger.error("Failed to create table PDF")
                print("[ERROR] Failed to create table PDF")
                return None
                
            original_pdf.insert_pdf(table_pdf, start_at=0)
            original_pdf.save(output_pdf_path)
            original_pdf.close()
            table_pdf.close()
            
            logger.info(f"PDF saved successfully to {output_pdf_path}")
            print(f"[INFO] PDF saved successfully to {output_pdf_path}")
            return buffer
            
        except Exception as e:
            logger.error(f"Error inserting table to PDF: {e}")
            print(f"[ERROR] Error inserting table to PDF: {e}")
            return None
    
    def index_pdf(self, output_pdf_path="indexed_pdf.pdf", pages=None):
        try:
            logger.info("Starting PDF indexing process")
            print("[INFO] Starting PDF indexing process")
            
            self.run_all_rois(pages)
            buffer = self.insert_dataframe_table_to_pdf(output_pdf_path)
            
            if buffer is None:
                logger.error("Failed to create indexed PDF")
                print("[ERROR] Failed to create indexed PDF")
                return None, [], [], [], []
            
            logger.info(f"Processed PDF saved to {output_pdf_path}")
            print(f"[INFO] Processed PDF saved to {output_pdf_path}")
            
            return buffer, self.rev_details, self.title_details, self.sheet_details, self.dnum_details
            
        except Exception as e:
            logger.error(f"Error in index_pdf: {e}")
            print(f"[ERROR] Error in index_pdf: {e}")
            return None, [], [], [], []