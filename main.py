from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
from typing import List, Dict, Any
from pydantic import BaseModel
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from pymongo import MongoClient
import certifi
from PDFDrawingProcessor import PDFDrawingProcessor
import tempfile
import os
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Technocrafts BOM to ERP Converter API",
    description="Convert BOM files to ERP format with intelligent item matching",
    root_path="/apii",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://abhistat.com",
        "https://www.abhistat.com",
        "http://abhistat.com",
        "http://www.abhistat.com",
        "https://technocraftserp.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Session-ID",
        "Cookie",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
)

class ItemMatch(BaseModel):
    item_code: str
    description: str
    make: str
    similarity_score: float
    confidence: str

class BOMProcessingResult(BaseModel):
    success: bool
    message: str
    total_items: int
    matched_items: int
    not_found_items: int
    processing_time: float
    results: List[Dict[str, Any]]

class ERPExportData(BaseModel):
    erp_data: List[Dict[str, Any]]
    not_found_data: List[Dict[str, Any]]

class ItemCodeUpdate(BaseModel):
    index: int
    itemCode: str

class UpdateItemCodesRequest(BaseModel):
    updates: List[ItemCodeUpdate]
    processed_bom: Dict[str, Any]

class BOMProcessor:
    def __init__(self):
        self.master_catalog = self._load_master_catalog()
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_master_catalog(self) -> pd.DataFrame:
        try:
            mongodb_uri = "mongodb+srv://atharav2601:Z3pFaAjfVFbnJD7K@technocrafts.z5vysrd.mongodb.net/erp?retryWrites=true&w=majority"
            client = MongoClient(mongodb_uri, tlsCAFile=certifi.where())
            db = client.erp
            collection = db.mastercatalogs
            
            documents = list(collection.find({}))
            
            if documents:
                df = pd.DataFrame(documents)
                
                column_mapping = {
                    'itemDescription': 'Item Description',
                    'family': 'Family',
                    'category': 'Category',
                    'itemCode': 'Item Code',
                    'make': 'Make',
                    'rating': 'Rating',
                    'otd': 'OTD',
                    'uom': 'UOM',
                    'exciseCategory': 'Excise Category',
                    'application': 'Application',
                    'salesDescription': 'Sales Description'
                }
                
                df = df.rename(columns=column_mapping)
                
                df['AC/DC'] = 'AC'
                
                required_columns = ['Item Code', 'Item Description', 'Make', 'Family', 'Category', 'AC/DC']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = ''
                
                df = df[required_columns]
                
                df = df.fillna('')
                
                logger.info(f"Loaded master catalog from MongoDB with {len(df)} items")
                return df
            else:
                logger.warning("No items found in MongoDB collection, creating empty catalog")
                return pd.DataFrame({
                    "Item Code": [],
                    "Item Description": [],
                    "Make": [],
                    "Family": [],
                    "Category": [],
                    "AC/DC": [],
                })
        except Exception as e:
            logger.error(f"Error loading master catalog from MongoDB: {e}")
            return pd.DataFrame({
                "Item Code": [],
                "Item Description": [],
                "Make": [],
                "Family": [],
                "Category": [],
                "AC/DC": [],
            })

    def extract_family(self, description: str) -> str:
        if not isinstance(description, str):
            return "OTHER"

        families = {
            "MCCB": ["MCCB", "MOLDED CASE CIRCUIT BREAKER"],
            "ACB": ["ACB", "AIR CIRCUIT BREAKER"],
            "MCB": ["MCB", "MINIATURE CIRCUIT BREAKER"],
            "CONTACTOR": ["CONTACTOR", "MAGNETIC CONTACTOR"],
            "RELAY": ["RELAY", "PROTECTION RELAY", "AUXILIARY RELAY"],
            "INDICATOR": ["INDICATOR", "LAMP", "PILOT LAMP", "LED"],
            "METER": ["METER", "AMMETER", "VOLTMETER", "ENERGY METER"],
            "TRANSFORMER": ["TRANSFORMER", "CT", "PT", "CURRENT TRANSFORMER"],
            "SWITCH": ["SWITCH", "SELECTOR SWITCH", "TOGGLE SWITCH"],
            "TERMINAL": ["TERMINAL", "TERMINAL BLOCK"],
            "CAPACITOR": ["CAPACITOR", "PF CAPACITOR"],
            "BUSBAR": ["BUSBAR", "BUS BAR"],
            "CABLE": ["CABLE", "WIRE", "CONDUCTOR"],
            "FUSE": ["FUSE", "HRC FUSE"],
        }

        description_upper = description.upper()
        for family, keywords in families.items():
            if any(keyword in description_upper for keyword in keywords):
                return family
        return "OTHER"

    def extract_category(self, description: str) -> str:
        if not isinstance(description, str):
            return "OTHER"

        categories = {
            "POWER DISTRIBUTION": ["DISTRIBUTION", "POWER", "MAIN"],
            "PROTECTION": ["PROTECTION", "BREAKER", "FUSE", "MCCB", "MCB", "ACB", "RCCB", "ELCB"],
            "CONTROL": ["CONTROL", "CONTACTOR", "RELAY", "PLC", "SWITCH", "BUTTON"],
            "METERING": ["METER", "MEASUREMENT", "AMMETER", "VOLTMETER"],
            "INDICATION": ["INDICATION", "INDICATOR", "LAMP", "LIGHT", "LED"],
            "CONNECTIVITY": ["TERMINAL", "CONNECTOR", "CABLE", "WIRE"],
            "TRANSFORMER": ["TRANSFORMER", "CT", "PT"],
        }

        description_upper = description.upper()
        for category, keywords in categories.items():
            if any(keyword in description_upper for keyword in keywords):
                return category
        return "OTHER"

    def extract_ac_dc(self, description: str) -> str:
        if not isinstance(description, str):
            return "AC"

        description_upper = description.upper()
        
        dc_terms = ["DC", " 24V", " 12V", " 48V", "DIRECT CURRENT"]
        ac_terms = ["AC", "230V", "415V", "440V", "ALTERNATING CURRENT"]
        
        if any(term in description_upper for term in dc_terms):
            return "DC"
        elif any(term in description_upper for term in ac_terms):
            return "AC"
        return "AC"

    def preprocess_description(self, description: str) -> str:
        if not isinstance(description, str):
            return ""
        
        text = description.upper()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def calculate_confidence(self, similarity_score: float, make_match: bool = False, family_match: bool = False) -> str:
        if similarity_score > 0.7 or (similarity_score > 0.5 and make_match):
            return "High"
        elif similarity_score > 0.4 or (similarity_score > 0.3 and family_match):
            return "Medium"
        else:
            return "Low"

    def verify_item_code(self, item_code: str) -> bool:
        if not item_code or pd.isna(item_code) or item_code == "NOT_FOUND":
            return False
        return item_code in self.master_catalog["Item Code"].values

    def find_similar_items(self, description: str, make: str, family: str, category: str, ac_dc: str) -> List[ItemMatch]:
        if not isinstance(description, str) or not description.strip():
            return []

        try:
            filtered_catalog = self.master_catalog.copy()
            
            if make and make.strip() and not filtered_catalog.empty:
                make_mask = filtered_catalog["Make"].str.upper() == make.upper()
                make_filtered = filtered_catalog[make_mask]
                if len(make_filtered) > 0:
                    filtered_catalog = make_filtered

            if family and family != "OTHER" and len(filtered_catalog) > 5:
                family_mask = filtered_catalog["Family"].str.upper() == family.upper()
                family_filtered = filtered_catalog[family_mask]
                if len(family_filtered) > 0:
                    filtered_catalog = family_filtered

            if category and category != "OTHER" and len(filtered_catalog) > 5:
                category_mask = filtered_catalog["Category"].str.upper() == category.upper()
                category_filtered = filtered_catalog[category_mask]
                if len(category_filtered) > 0:
                    filtered_catalog = category_filtered

            if len(filtered_catalog) == 0:
                filtered_catalog = self.master_catalog

            if filtered_catalog.empty:
                return []

            bom_description = self.preprocess_description(description)
            catalog_descriptions = [self.preprocess_description(desc) for desc in filtered_catalog["Item Description"]]
            
            all_descriptions = catalog_descriptions + [bom_description]
            
            if len(all_descriptions) < 2:
                return []

            tfidf_matrix = self.vectorizer.fit_transform(all_descriptions)
            cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            top_indices = np.argsort(cosine_similarities)[::-1][:3]
            
            similar_items = []
            for idx in top_indices:
                if cosine_similarities[idx] > 0.1:
                    item = filtered_catalog.iloc[idx]
                    make_match = str(item.get("Make", "")).upper() == make.upper() if make else False
                    family_match = str(item.get("Family", "")).upper() == family.upper() if family != "OTHER" else False
                    
                    confidence = self.calculate_confidence(
                        cosine_similarities[idx], 
                        make_match=make_match,
                        family_match=family_match
                    )
                    
                    similar_items.append(ItemMatch(
                        item_code=str(item["Item Code"]),
                        description=str(item["Item Description"]),
                        make=str(item.get("Make", "")),
                        similarity_score=float(cosine_similarities[idx]),
                        confidence=confidence
                    ))
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            return []

    async def process_bom_item(self, row: pd.Series) -> Dict[str, Any]:
        description = str(row.get("DESCRIPTION", ""))
        make = str(row.get("MAKE", ""))
        qty = row.get("TOTAL QTY 2 SETS", 1)
        type_ref = str(row.get("TYPE REFERENCE", ""))
        existing_item_code = str(row.get("ITEM CODE", "")) if "ITEM CODE" in row else ""
        
        family = self.extract_family(description)
        category = self.extract_category(description)
        ac_dc = self.extract_ac_dc(description)
        
        result = {
            "original_description": description,
            "make": make,
            "quantity": qty,
            "type_reference": type_ref,
            "family": family,
            "category": category,
            "ac_dc": ac_dc,
            "existing_item_code": existing_item_code,
            "matched_item_code": "NOT_FOUND",
            "confidence": "Not Found",
            "similar_items": [],
            "status": "not_found"
        }
        
        if existing_item_code and existing_item_code != "NOT_FOUND":
            if self.verify_item_code(existing_item_code):
                result.update({
                    "matched_item_code": existing_item_code,
                    "confidence": "Existing",
                    "status": "existing_verified"
                })
                return result
            else:
                similar_items = self.find_similar_items(description, make, family, category, ac_dc)
        else:
            similar_items = self.find_similar_items(description, make, family, category, ac_dc)
        
        if similar_items:
            result.update({
                "matched_item_code": similar_items[0].item_code,
                "confidence": similar_items[0].confidence,
                "similar_items": [item.dict() for item in similar_items],
                "status": "matched"
            })
        
        return result

processor = BOMProcessor()

@app.post("/process-bom", response_model=BOMProcessingResult)
async def process_bom(file: UploadFile = File(...)):
    start_time = datetime.now()
    logger.info(f"Received BOM file: {file.filename}")
    print(f"[INFO] Received BOM file: {file.filename}")

    if not file.filename.endswith(('.xlsx', '.xls')):
        logger.error("File must be an Excel file (.xlsx or .xls)")
        print("[ERROR] File must be an Excel file (.xlsx or .xls)")
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    try:
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from uploaded file.")
        print(f"[INFO] Read {len(contents)} bytes from uploaded file.")
        bom_data = pd.read_excel(io.BytesIO(contents))
        
        bom_data.columns = [col.strip().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ') for col in bom_data.columns]
        required_columns = ["DESCRIPTION", "MAKE", "TYPE REFERENCE", "TOTAL QTY 2 SETS"]
        missing_columns = [col for col in required_columns if col not in bom_data.columns]
        
        if missing_columns:
            available_columns = list(bom_data.columns)
            logger.error(f"Missing required columns: {missing_columns}. Available columns: {available_columns}")
            print(f"[ERROR] Missing required columns: {missing_columns}. Available columns: {available_columns}")
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Available columns: {available_columns}"
            )
        
        results = []
        tasks = []
        
        for idx, row in bom_data.iterrows():
            logger.info(f"Processing row {idx+1}/{len(bom_data)}")
            print(f"[INFO] Processing row {idx+1}/{len(bom_data)}")
            task = processor.process_bom_item(row)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        matched_count = sum(1 for r in results if r["status"] in ["matched", "existing_verified"])
        not_found_count = len(results) - matched_count
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processed {len(results)} items: {matched_count} matched, {not_found_count} not found in {processing_time:.2f}s")
        print(f"[INFO] Processed {len(results)} items: {matched_count} matched, {not_found_count} not found in {processing_time:.2f}s")
        
        return BOMProcessingResult(
            success=True,
            message=f"Successfully processed {len(results)} items. {matched_count} matched, {not_found_count} not found.",
            total_items=len(results),
            matched_items=matched_count,
            not_found_items=not_found_count,
            processing_time=processing_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error processing BOM: {e}")
        print(f"[ERROR] Error processing BOM: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing BOM file: {str(e)}")

@app.post("/update-item-codes")
async def update_item_codes(request: UpdateItemCodesRequest):
    try:
        processed_bom = request.processed_bom
        updates = request.updates
        
        items = processed_bom.get("items", [])
        summary = processed_bom.get("summary", {})
        
        for update in updates:
            if 0 <= update.index < len(items):
                item = items[update.index]
                old_item_code = item.get("matched_item_code", "")
                new_item_code = update.itemCode
                
                if processor.verify_item_code(new_item_code):
                    item["matched_item_code"] = new_item_code
                    item["confidence"] = "High"
                    item["status"] = "matched"
                    
                    catalog_item = processor.master_catalog[
                        processor.master_catalog["Item Code"] == new_item_code
                    ].iloc[0] if len(processor.master_catalog[
                        processor.master_catalog["Item Code"] == new_item_code
                    ]) > 0 else None
                    
                    if catalog_item is not None:
                        item["similar_items"] = [{
                            "item_code": new_item_code,
                            "description": str(catalog_item.get("Item Description", "")),
                            "make": str(catalog_item.get("Make", "")),
                            "similarity_score": 1.0,
                            "confidence": "High"
                        }]
                else:
                    item["matched_item_code"] = new_item_code
                    item["confidence"] = "Medium"
                    item["status"] = "updated"
        
        updated_matched = sum(1 for item in items if item["status"] in ["matched", "existing_verified", "updated"])
        updated_not_found = len(items) - updated_matched
        updated_existing = sum(1 for item in items if item["status"] == "existing_verified")
        
        updated_summary = {
            "total": len(items),
            "matched": updated_matched - updated_existing,
            "notFound": updated_not_found,
            "existing": updated_existing
        }
        
        return {
            "success": True,
            "message": f"Successfully updated {len(updates)} item codes",
            "items": items,
            "summary": updated_summary
        }
        
    except Exception as e:
        logger.error(f"Error updating item codes: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating item codes: {str(e)}")

@app.post("/export-erp")
async def export_erp(processing_result: Dict[str, Any]):
    try:
        results = processing_result.get("results", [])
        
        erp_data = []
        not_found_data = []
        
        for result in results:
            base_data = {
                "Original Description": result["original_description"],
                "Make": result["make"],
                "Quantity": result["quantity"],
                "Type Reference": result["type_reference"],
                "Family": result["family"],
                "Category": result["category"],
                "AC/DC": result["ac_dc"],
            }
            
            if result["status"] in ["matched", "existing_verified"]:
                erp_record = {
                    "Item Code": result["matched_item_code"],
                    "Feeder Name": 1,
                    "Feeder Qty": 1,
                    "Feeder Description": "",
                    "Qty/Feeder": result["quantity"],
                    "UOM": "",
                    "Designation": result["type_reference"],
                    "Remark": "",
                    "ListPrice": "",
                    "Discount": "",
                    "Supply Type": "",
                }
                erp_data.append(erp_record)
            else:
                not_found_record = {
                    "Item Code": "NOT_FOUND",
                    "Confidence": result["confidence"],
                    "Similar Items": str([item["item_code"] for item in result.get("similar_items", [])]),
                    **base_data
                }
                not_found_data.append(not_found_record)
        
        erp_excel = io.BytesIO()
        not_found_excel = io.BytesIO()
        
        if erp_data:
            erp_df = pd.DataFrame(erp_data)
            erp_df.to_excel(erp_excel, index=False, sheet_name="ERP Data")
        
        if not_found_data:
            not_found_df = pd.DataFrame(not_found_data)
            not_found_df.to_excel(not_found_excel, index=False, sheet_name="Not Found Items")
        
        erp_excel.seek(0)
        not_found_excel.seek(0)
        
        return JSONResponse({
            "success": True,
            "message": f"Generated ERP export with {len(erp_data)} matched items and {len(not_found_data)} not found items",
            "erp_items_count": len(erp_data),
            "not_found_count": len(not_found_data),
        })
        
    except Exception as e:
        logger.error(f"Error exporting ERP data: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting ERP data: {str(e)}")

@app.get("/download-erp/{file_type}")
async def download_erp_file(file_type: str, processing_result: Dict[str, Any]):
    if file_type not in ["erp", "not_found"]:
        raise HTTPException(status_code=400, detail="File type must be 'erp' or 'not_found'")
    
    try:
        results = processing_result.get("results", [])
        
        if file_type == "erp":
            erp_data = []
            for result in results:
                if result["status"] in ["matched", "existing_verified"]:
                    erp_record = {
                        "Item Code": result["matched_item_code"],
                        "Feeder Name": 1,
                        "Feeder Qty": 1,
                        "Feeder Description": "",
                        "Qty/Feeder": result["quantity"],
                        "UOM": "",
                        "Designation": result["type_reference"],
                        "Remark": "",
                        "ListPrice": "",
                        "Discount": "",
                        "Supply Type": "",
                    }
                    erp_data.append(erp_record)
            
            df = pd.DataFrame(erp_data)
            filename = "erp_export.xlsx"
        
        else:
            not_found_data = []
            for result in results:
                if result["status"] == "not_found":
                    not_found_record = {
                        "Original Description": result["original_description"],
                        "Make": result["make"],
                        "Quantity": result["quantity"],
                        "Type Reference": result["type_reference"],
                        "Family": result["family"],
                        "Category": result["category"],
                        "Similar Items": ", ".join([item["item_code"] for item in result.get("similar_items", [])]),
                    }
                    not_found_data.append(not_found_record)
            
            df = pd.DataFrame(not_found_data)
            filename = "not_found_items.xlsx"
        
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(excel_buffer.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/master-catalog-info")
async def get_master_catalog_info():
    try:
        catalog_info = {
            "total_items": len(processor.master_catalog),
            "unique_makes": processor.master_catalog["Make"].nunique() if "Make" in processor.master_catalog.columns else 0,
            "families": processor.master_catalog["Family"].value_counts().to_dict() if "Family" in processor.master_catalog.columns else {},
            "categories": processor.master_catalog["Category"].value_counts().to_dict() if "Category" in processor.master_catalog.columns else {},
        }
        return catalog_info
    except Exception as e:
        logger.error(f"Error getting catalog info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving catalog information: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "master_catalog_loaded": len(processor.master_catalog) > 0,
        "master_catalog_items": len(processor.master_catalog)
    }

class PDFProcessingResult(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    processing_time: float
    
def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj

def buffer_to_base64(buffer):
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_buffer(b64_string):
    return io.BytesIO(base64.b64decode(b64_string))

@app.post("/process-pdf", response_model=PDFProcessingResult)
async def process_pdf(file: UploadFile = File(...)):
    start_time = datetime.now()
    logger.info(f"Received PDF file: {file.filename}")
    print(f"[INFO] Received PDF file: {file.filename}")

    if not file.filename.lower().endswith('.pdf'):
        logger.error("File must be a PDF file")
        print("[ERROR] File must be a PDF file")
        raise HTTPException(status_code=400, detail="File must be a PDF file")

    temp_file_path = None
    try:
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from uploaded PDF")
        print(f"[INFO] Read {len(contents)} bytes from uploaded PDF")
        
        if len(contents) == 0:
            logger.error("Uploaded file is empty")
            print("[ERROR] Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        processor_instance = PDFDrawingProcessor(temp_file_path)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='_indexed.pdf') as output_file:
            output_path = output_file.name

        result = processor_instance.index_pdf(output_path)
        
        if result[0] is None:
            logger.error("PDF processing failed")
            print("[ERROR] PDF processing failed")
            raise HTTPException(status_code=500, detail="PDF processing failed")
            
        buffer, rev_details, title_details, sheet_details, dnum_details = result

        df = processor_instance.get_dataframe()
        
        data = {
            "table_data": df.to_dict(orient='records') if not df.empty else [],
            "total_pages": len(processor_instance.doc) if hasattr(processor_instance, 'doc') else 0,
            "processed_pages": len(df) if not df.empty else 0,
            "rev_details": [
                (
                    int(page),
                    texts.tolist() if isinstance(texts, np.ndarray) else texts,
                    [float(s) for s in (scores.tolist() if isinstance(scores, np.ndarray) else scores)],
                    polys.tolist() if isinstance(polys, np.ndarray) else polys
                )
                for page, texts, scores, polys in rev_details
            ],
            "title_details": [
                (
                    int(page),
                    texts.tolist() if isinstance(texts, np.ndarray) else texts,
                    [float(s) for s in (scores.tolist() if isinstance(scores, np.ndarray) else scores)],
                    polys.tolist() if isinstance(polys, np.ndarray) else polys
                )
                for page, texts, scores, polys in title_details
            ],
            "sheet_details": [
                (
                    int(page),
                    texts.tolist() if isinstance(texts, np.ndarray) else texts,
                    [float(s) for s in (scores.tolist() if isinstance(scores, np.ndarray) else scores)],
                    polys.tolist() if isinstance(polys, np.ndarray) else polys
                )
                for page, texts, scores, polys in sheet_details
            ],
            "dnum_details": [
                (
                    int(page),
                    texts.tolist() if isinstance(texts, np.ndarray) else texts,
                    [float(s) for s in (scores.tolist() if isinstance(scores, np.ndarray) else scores)],
                    polys.tolist() if isinstance(polys, np.ndarray) else polys
                )
                for page, texts, scores, polys in dnum_details
            ],
            "processed_pdf_base64": buffer_to_base64(buffer)
        }
        data = convert_ndarray_to_list(data)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processed PDF successfully in {processing_time:.2f}s")
        print(f"[INFO] Processed PDF successfully in {processing_time:.2f}s")

        return PDFProcessingResult(
            success=True,
            message=f"Successfully processed PDF with {data['total_pages']} pages",
            data=data,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        print(f"[ERROR] Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
                
class PDFBase64Request(BaseModel):
    processed_pdf_base64: str

@app.post("/download-processed-pdf")
async def download_processed_pdf(request: PDFBase64Request):
    try:
        buffer = base64_to_buffer(request.processed_pdf_base64)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=processed.pdf"}
        )
    except Exception as e:
        logger.error(f"Error downloading processed PDF: {str(e)}")
        print(f"[ERROR] Error downloading processed PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading processed PDF: {str(e)}")