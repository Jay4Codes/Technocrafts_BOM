import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
import os

st.set_page_config(page_title="Technocrafts BOM to ERP Converter", layout="wide")

if os.path.exists("master_catalog.xlsx"):
    master_catalog = pd.read_excel("master_catalog.xlsx")
else:
    master_catalog = pd.DataFrame({
        "Item Code": [],
        "Item Description": [],
        "Make": [],
        "Family": [],
        "Category": [],
        "AC/DC": [],
    })

def extract_family(description):
    if not isinstance(description, str):
        return ""

    families = {
        "MCCB": ["MCCB", "Molded Case Circuit Breaker"],
        "ACB": ["ACB", "Air Circuit Breaker"],
        "MCB": ["MCB", "Miniature Circuit Breaker"],
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
        for keyword in keywords:
            if keyword.upper() in description_upper:
                return family

    return "OTHER"

def extract_category(description):
    if not isinstance(description, str):
        return ""

    categories = {
        "POWER DISTRIBUTION": ["DISTRIBUTION", "POWER", "MAIN"],
        "PROTECTION": [
            "PROTECTION",
            "BREAKER",
            "FUSE",
            "MCCB",
            "MCB",
            "ACB",
            "RCCB",
            "ELCB",
        ],
        "CONTROL": ["CONTROL", "CONTACTOR", "RELAY", "PLC", "SWITCH", "BUTTON"],
        "METERING": ["METER", "MEASUREMENT", "AMMETER", "VOLTMETER"],
        "INDICATION": ["INDICATION", "INDICATOR", "LAMP", "LIGHT", "LED"],
        "CONNECTIVITY": ["TERMINAL", "CONNECTOR", "CABLE", "WIRE"],
        "TRANSFORMER": ["TRANSFORMER", "CT", "PT"],
    }

    description_upper = description.upper()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.upper() in description_upper:
                return category

    return "OTHER"

def extract_ac_dc(description):
    if not isinstance(description, str):
        return "AC"

    description_upper = description.upper()

    if any(
        term in description_upper
        for term in ["DC", " 24V", " 12V", " 48V", "DIRECT CURRENT"]
    ):
        return "DC"

    if any(
        term in description_upper
        for term in ["AC", "230V", "415V", "440V", "ALTERNATING CURRENT"]
    ):
        return "AC"

    return "AC"

def preprocess_description(description):
    if not isinstance(description, str):
        return ""

    text = description.upper()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def find_similar_items(row, master_catalog):
    """Find the top 5 similar items in the master catalog based on description and make"""
    if not isinstance(row["DESCRIPTION"], str) or pd.isna(row["DESCRIPTION"]):
        return []

    description = row["DESCRIPTION"]
    family = row["FAMILY"]
    category = row["CATEGORY"]
    ac_dc = row["AC/DC"]
    make = row["MAKE"] if "MAKE" in row and isinstance(row["MAKE"], str) else ""

    filtered_catalog = master_catalog.copy()
    
    if make and make.strip():
        make_filtered = filtered_catalog[
            filtered_catalog["Make"].str.upper() == make.upper()
            if pd.notna(filtered_catalog["Make"]).all()
            else False
        ]
        if len(make_filtered) > 0:
            filtered_catalog = make_filtered

    
    if family and family != "OTHER" and len(filtered_catalog) > 5:
        family_filtered = filtered_catalog[
            filtered_catalog["Family"].str.upper() == family.upper()
            if pd.notna(filtered_catalog["Family"]).all()
            else False
        ]
        if len(family_filtered) > 0:
            filtered_catalog = family_filtered

    
    if category and category != "OTHER" and len(filtered_catalog) > 5:
        category_filtered = filtered_catalog[
            filtered_catalog["Category"].str.upper() == category.upper()
            if pd.notna(filtered_catalog["Category"]).all()
            else False
        ]
        if len(category_filtered) > 0:
            filtered_catalog = category_filtered

    
    if len(filtered_catalog) == 0:
        filtered_catalog = master_catalog

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        
        bom_description = preprocess_description(description)
        catalog_descriptions = [preprocess_description(desc) for desc in filtered_catalog["Item Description"]]
        
        
        all_descriptions = catalog_descriptions + [bom_description]
        
        
        tfidf_matrix = vectorizer.fit_transform(all_descriptions)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
        
        
        top_indices = np.argsort(cosine_similarities)[::-1][:5]
        
        
        similar_items = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0.2:  
                item = filtered_catalog.iloc[idx]
                similar_items.append({
                    "Item Code": item["Item Code"],
                    "Description": item["Item Description"],
                    "Make": item["Make"] if "Make" in item else "",
                    "Similarity": cosine_similarities[idx]
                })
        
        return similar_items
    except Exception as e:
        st.error(f"Error in similarity calculation: {e}")
        return []


def verify_item_code(item_code, master_catalog):
    """Verify if an item code exists in the master catalog"""
    if not item_code or pd.isna(item_code) or item_code == "NOT_FOUND":
        return False
    
    return item_code in master_catalog["Item Code"].values

def calculate_confidence(similarity_score, make_match=False, family_match=False):
    """Calculate confidence level based on similarity score and other matches"""
    if similarity_score > 0.7 or (similarity_score > 0.5 and make_match):
        return "High"
    elif similarity_score > 0.4 or (similarity_score > 0.3 and family_match):
        return "Medium"
    else:
        return "Low"

def update_master_catalog(new_catalog):
    """Update master catalog with new items from uploaded catalog"""
    global master_catalog
    
    required_columns = ["Item Code", "Item Description", "Make", "Family", "Category", "AC/DC"]
    
    for col in required_columns:
        if col not in master_catalog.columns:
            master_catalog[col] = ""
        if col not in new_catalog.columns:
            new_catalog[col] = ""
    
    current_item_codes = set(master_catalog["Item Code"].values)
    new_items = new_catalog[~new_catalog["Item Code"].isin(current_item_codes)]
    
    if len(new_items) > 0:
        master_catalog = pd.concat([master_catalog, new_items[required_columns]], ignore_index=True)
        master_catalog.to_excel("master_catalog.xlsx", index=False)
        
        return len(new_items)
    
    return 0

def main():
    st.title("Technocrafts BOM to ERP Format Converter")
    
    st.sidebar.header("Upload Files")
    
    uploaded_bom = st.sidebar.file_uploader("Upload BOM Excel File", type=["xlsx", "xls"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optional: Update Master Catalog")
    uploaded_catalog = st.sidebar.file_uploader("Upload New Master Catalog", type=["xlsx", "xls"])
    
    
    if uploaded_catalog:
        try:
            new_catalog = pd.read_excel(uploaded_catalog)
            if "Item Code" in new_catalog.columns and "Item Description" in new_catalog.columns:
                new_items_count = update_master_catalog(new_catalog)
                st.sidebar.success(f"Master catalog updated with {new_items_count} new items")
            else:
                st.sidebar.error("Uploaded catalog must contain 'Item Code' and 'Item Description' columns")
        except Exception as e:
            st.sidebar.error(f"Error updating master catalog: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Current master catalog contains {len(master_catalog)} items")
    
    if uploaded_bom:
        try:
            bom_data = pd.read_excel(uploaded_bom)
            
            st.subheader("BOM File Preview")
            st.dataframe(bom_data.head())
            
            
            st.subheader("Column Mapping")
            
            col1, col2 = st.columns(2)
            with col1:
                description_col = st.selectbox(
                    "Description Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("DESCRIPTION")
                        if "DESCRIPTION" in bom_data.columns
                        else 0
                    ),
                )
                
                make_col = st.selectbox(
                    "Make Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("MAKE")
                        if "MAKE" in bom_data.columns
                        else 0
                    ),
                )
                
            with col2:
                type_ref_col = st.selectbox(
                    "Type Reference Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("TYPE REFERENCE")
                        if "TYPE REFERENCE" in bom_data.columns
                        else 0
                    ),
                )
                
                
                qty_cols = [col for col in bom_data.columns if "QTY" in col.upper()]
                default_qty_idx = 0
                if qty_cols:
                    default_qty_idx = bom_data.columns.get_loc(qty_cols[0])
                
                qty_col = st.selectbox(
                    "Quantity Column",
                    options=bom_data.columns,
                    index=default_qty_idx
                )
            
            item_code_exists = "ITEM CODE" in bom_data.columns
            for col in bom_data.columns:
                if "ITEM" in col.upper() and "CODE" in col.upper():
                    item_code_exists = True
                    break
            
            if item_code_exists:
                item_code_col = st.selectbox(
                    "Item Code Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("ITEM CODE")
                        if "ITEM CODE" in bom_data.columns
                        else next((i for i, col in enumerate(bom_data.columns) 
                                  if "ITEM" in col.upper() and "CODE" in col.upper()), 0)
                    ),
                )
            else:
                st.info("No Item Code column detected in the uploaded BOM. A new column will be created.")
                item_code_col = None
            
            if st.button("Process BOM"):
                
                bom_mapped = bom_data.copy()
                bom_mapped.rename(
                    columns={
                        description_col: "DESCRIPTION",
                        make_col: "MAKE",
                        qty_col: "QTY",
                        type_ref_col: "TYPE_REF",
                    },
                    inplace=True,
                )
                
                if item_code_col:
                    bom_mapped.rename(columns={item_code_col: "ITEM_CODE"}, inplace=True)
                else:
                    bom_mapped["ITEM_CODE"] = "NOT_FOUND"
                
                with st.spinner("Extracting features from descriptions..."):
                    bom_mapped["FAMILY"] = bom_mapped["DESCRIPTION"].apply(extract_family)
                    bom_mapped["CATEGORY"] = bom_mapped["DESCRIPTION"].apply(extract_category)
                    bom_mapped["AC/DC"] = bom_mapped["DESCRIPTION"].apply(extract_ac_dc)
                 
                with st.spinner("Processing BOM items..."):
                    
                    item_code_options = {}
                    
                    
                    progress_bar = st.progress(0)
                    total_rows = len(bom_mapped)
                    
                    for idx, row in bom_mapped.iterrows():
                        
                        progress_bar.progress((idx + 1) / total_rows)
                        
                        
                        if "ITEM_CODE" in row and row["ITEM_CODE"] and row["ITEM_CODE"] != "NOT_FOUND":
                            
                            if verify_item_code(row["ITEM_CODE"], master_catalog):
                                
                                item_code_options[idx] = [row["ITEM_CODE"]]
                                
                                bom_mapped.at[idx, "CONFIDENCE"] = "Existing"
                            else:
                                
                                similar_items = find_similar_items(row, master_catalog)
                                options = [item["Item Code"] for item in similar_items]
                                options.append("NOT_FOUND")
                                item_code_options[idx] = options
                                
                                
                                bom_mapped.at[idx, "ITEM_CODE"] = "NOT_FOUND"
                                bom_mapped.at[idx, "CONFIDENCE"] = "Not Found"
                        
                        else:
                            
                            similar_items = find_similar_items(row, master_catalog)
                            
                            if similar_items:
                                options = [item["Item Code"] for item in similar_items]
                                options.append("NOT_FOUND")
                                item_code_options[idx] = options
                                
                                
                                bom_mapped.at[idx, "ITEM_CODE"] = similar_items[0]["Item Code"]
                                
                                
                                make_match = similar_items[0].get("Make", "").upper() == row["MAKE"].upper() if isinstance(row["MAKE"], str) else False
                                family_match = False  
                                
                                confidence = calculate_confidence(
                                    similar_items[0]["Similarity"], 
                                    make_match=make_match,
                                    family_match=family_match
                                )
                                bom_mapped.at[idx, "CONFIDENCE"] = confidence
                            else:
                                item_code_options[idx] = ["NOT_FOUND"]
                                bom_mapped.at[idx, "ITEM_CODE"] = "NOT_FOUND"
                                bom_mapped.at[idx, "CONFIDENCE"] = "Not Found"
                        
                    progress_bar.empty()
                
                st.subheader("Enhanced BOM with Item Code Selection")
                
                interactive_bom = st.empty()
                interactive_bom.dataframe(bom_mapped)
                
                with st.form("item_code_form"):
                    st.write("Select Item Codes for BOM Items")
                    
                    row_updates = {}
                    
                    for idx, options in item_code_options.items():
                        if len(options) > 1:
                            row_desc = bom_mapped.loc[idx, "DESCRIPTION"][:50] + "..." if len(bom_mapped.loc[idx, "DESCRIPTION"]) > 50 else bom_mapped.loc[idx, "DESCRIPTION"]
                            current_value = bom_mapped.loc[idx, "ITEM_CODE"]
                            selected_item_code = st.selectbox(
                                f"Row {idx+1}: {row_desc}",
                                options=options,
                                index=options.index(current_value) if current_value in options else 0
                            )
                            row_updates[idx] = selected_item_code
                    
                    submit_button = st.form_submit_button("Update Item Codes")
                    
                    if submit_button:
                        
                        for idx, item_code in row_updates.items():
                            bom_mapped.at[idx, "ITEM_CODE"] = item_code
                            
                            
                            if item_code == "NOT_FOUND":
                                bom_mapped.at[idx, "CONFIDENCE"] = "Not Found"
                            else:
                                
                                similar_items = find_similar_items(bom_mapped.loc[idx], master_catalog)
                                selected_item = next((item for item in similar_items if item["Item Code"] == item_code), None)
                                
                                if selected_item:
                                    make_match = selected_item.get("Make", "").upper() == bom_mapped.loc[idx, "MAKE"].upper() if isinstance(bom_mapped.loc[idx, "MAKE"], str) else False
                                    confidence = calculate_confidence(selected_item["Similarity"], make_match=make_match)
                                    bom_mapped.at[idx, "CONFIDENCE"] = confidence
                             
                        interactive_bom.dataframe(bom_mapped)
                
                erp_data = pd.DataFrame({
                    "Item Code": bom_mapped["ITEM_CODE"],
                    "Feeder Name": 1,
                    "Feeder Qty": 1,
                    "Feeder Description": "",
                    "Qty/Feeder": bom_mapped["QTY"],
                    "UOM": "",
                    "Designation": bom_mapped["TYPE_REF"],
                    "Remark": "",
                    "ListPrice": "",
                    "Discount": "",
                    "Supply Type": "",
                })
                
                valid_erp_data = erp_data[erp_data["Item Code"] != "NOT_FOUND"]
                not_found_bom = bom_mapped[bom_mapped["ITEM_CODE"] == "NOT_FOUND"]
                
                st.subheader("Final ERP Excel")
                st.dataframe(valid_erp_data)
                
                st.subheader("Items Not Found in Master Catalog")
                st.dataframe(not_found_bom)
                
                updated_bom_excel = io.BytesIO()
                bom_mapped.to_excel(updated_bom_excel, index=False)
                updated_bom_excel.seek(0)
                
                erp_excel = io.BytesIO()
                valid_erp_data.to_excel(erp_excel, index=False)
                erp_excel.seek(0)
                
                not_found_excel = io.BytesIO()
                not_found_bom.to_excel(not_found_excel, index=False)
                not_found_excel.seek(0)
                
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="Download Updated BOM",
                        data=updated_bom_excel,
                        file_name="updated_bom.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
                with col2:
                    st.download_button(
                        label="Download ERP Excel",
                        data=erp_excel,
                        file_name="final_erp.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
                with col3:
                    st.download_button(
                        label="Download Not Found Items",
                        data=not_found_excel,
                        file_name="not_found_items.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload the BOM Excel file to continue.")
        
        st.subheader("Expected BOM File Format")
        sample_bom = pd.DataFrame({
            "SR. NO.": [1, 2, 3],
            "DESCRIPTION": [
                "MCCB 3P 100A 36kA TM",
                "Contactor 3P 95A 415V AC",
                "LED Indicator Light 230V AC Green",
            ],
            "MAKE": ["Schneider", "ABB", "Siemens"],
            "TOTAL QTY 2 SETS": [2, 4, 8],
            "TYPE REFERENCE": ["Q1", "K1", "L1"],
            "ITEM CODE": ["MC-100-36", "", "IND-LED-GRN"],  
        })
        st.dataframe(sample_bom)
        
        st.subheader("Expected Master Catalog Format")
        sample_catalog = pd.DataFrame({
            "Item Code": ["MC-100-36", "CON-95-AC", "IND-LED-GRN"],
            "Item Description": [
                "MCCB 3P 100A 36kA Thermal Magnetic",
                "Contactor 3 Pole 95A 415V AC Coil",
                "LED Indicator Lamp 230V AC Green"
            ],
            "Make": ["Schneider", "ABB", "Siemens"],
            "Family": ["MCCB", "CONTACTOR", "INDICATOR"],
            "Category": ["PROTECTION", "CONTROL", "INDICATION"],
            "AC/DC": ["AC", "AC", "AC"],
        })
        st.dataframe(sample_catalog)


if __name__ == "__main__":
    main()