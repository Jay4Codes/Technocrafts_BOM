import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

st.set_page_config(page_title="Technocrafts BOM to ERP Converter", layout="wide")

master_catalog = pd.read_excel("master_catalog.xlsx")


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


def calculate_description_similarity(bom_description, catalog_description):
    if not isinstance(bom_description, str) or not isinstance(catalog_description, str):
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform([bom_description, catalog_description])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0.0


def verify_item_code_and_get_confidence(row, master_catalog):
    item_code = row["ITEM_CODE"]
    description = row["DESCRIPTION"] if isinstance(row["DESCRIPTION"], str) else ""
    make = row["MAKE"] if "MAKE" in row and isinstance(row["MAKE"], str) else ""
    family = row["FAMILY"]
    category = row["CATEGORY"]

    # Check if item code exists in master catalog
    matching_items = master_catalog[master_catalog["Item Code"] == item_code]

    if len(matching_items) == 0:
        return {
            "Match Found": False,
            "Item Code": item_code,
            "Match Description": "",
            "Confidence": "Not Found",
            "Similarity Score": 0.0,
        }

    # Item code found in master catalog
    matched_item = matching_items.iloc[0]

    # Calculate similarity between descriptions
    bom_desc_processed = preprocess_description(description)
    catalog_desc_processed = preprocess_description(matched_item["Item Description"])
    similarity_score = calculate_description_similarity(
        bom_desc_processed, catalog_desc_processed
    )

    # Make matching (if available)
    make_match = False
    if make and isinstance(matched_item["Make"], str):
        make_match = make.upper() == matched_item["Make"].upper()

    # Family matching (if available)
    family_match = False
    if isinstance(matched_item["Family"], str):
        family_match = family.upper() == matched_item["Family"].upper()

    # Category matching (if available)
    category_match = False
    if isinstance(matched_item["Category"], str):
        category_match = category.upper() == matched_item["Category"].upper()

    # Determine confidence level
    confidence = "Low"
    if similarity_score > 0.7:
        confidence = "High"
    elif similarity_score > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Boost confidence if make or family matches
    if make_match and (confidence == "Medium" or similarity_score > 0.3):
        confidence = "High"
    if family_match and confidence == "Low" and similarity_score > 0.2:
        confidence = "Medium"

    return {
        "Match Found": True,
        "Item Code": item_code,
        "Match Description": matched_item["Item Description"],
        "Confidence": confidence,
        "Similarity Score": similarity_score,
    }


def main():
    st.title("Technocrafts BOM to ERP Format Converter")

    with st.sidebar:
        st.header("Upload Files")
        uploaded_bom = st.file_uploader("Upload BOM Excel File", type=["xlsx", "xls"])

    if uploaded_bom:
        try:
            bom_data = pd.read_excel(uploaded_bom)

            st.subheader("BOM File Preview")
            st.dataframe(bom_data.head())

            # st.subheader("Master Catalog Preview")
            # st.dataframe(master_catalog.head())

            st.subheader("Column Mapping")

            col1, col2, col3, col4, col5 = st.columns(5)

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

            with col2:
                make_col = st.selectbox(
                    "Make Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("MAKE")
                        if "MAKE" in bom_data.columns
                        else 0
                    ),
                )

            with col3:
                qty_col = st.selectbox(
                    "Quantity Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("TOTAL QTY 2 SETS")
                        if "TOTAL QTY 2 SETS" in bom_data.columns
                        else 0
                    ),
                )

            with col4:
                type_ref_col = st.selectbox(
                    "Type Reference Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("TYPE REFERENCE")
                        if "TYPE REFERENCE" in bom_data.columns
                        else 0
                    ),
                )

            with col5:
                item_code_col = st.selectbox(
                    "Item Code Column",
                    options=bom_data.columns,
                    index=(
                        bom_data.columns.get_loc("ITEM CODE")
                        if "ITEM CODE" in bom_data.columns
                        else 0
                    ),
                )

            if st.button("Process BOM"):
                bom_mapped = bom_data.copy()
                bom_mapped.rename(
                    columns={
                        description_col: "DESCRIPTION",
                        make_col: "MAKE",
                        qty_col: "QTY",
                        type_ref_col: "TYPE_REF",
                        item_code_col: "ITEM_CODE",
                    },
                    inplace=True,
                )

                with st.spinner("Extracting features from descriptions..."):
                    bom_mapped["FAMILY"] = bom_mapped["DESCRIPTION"].apply(
                        extract_family
                    )
                    bom_mapped["CATEGORY"] = bom_mapped["DESCRIPTION"].apply(
                        extract_category
                    )
                    bom_mapped["AC/DC"] = bom_mapped["DESCRIPTION"].apply(extract_ac_dc)

                with st.spinner("Verifying item codes with master catalog..."):
                    verification_results = []
                    for idx, row in bom_mapped.iterrows():
                        result = verify_item_code_and_get_confidence(
                            row, master_catalog
                        )
                        result["Index"] = idx
                        verification_results.append(result)

                verification_df = pd.DataFrame(verification_results)

                # Create ERP data using the provided item codes
                erp_data = pd.DataFrame(
                    {
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
                    }
                )

                st.subheader("Analysis Results")

                st.write("Feature Extraction from Descriptions")
                feature_df = bom_mapped[["DESCRIPTION", "FAMILY", "CATEGORY", "AC/DC"]]
                st.dataframe(feature_df)

                st.write("Item Code Verification Results")
                verification_display = verification_df[
                    [
                        "Item Code",
                        "Match Found",
                        "Match Description",
                        "Confidence",
                        "Similarity Score",
                    ]
                ]
                st.dataframe(verification_display)

                # Count confidence levels
                confidence_counts = (
                    verification_df["Confidence"].value_counts().to_dict()
                )
                st.write("Confidence Summary:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Confidence", confidence_counts.get("High", 0))
                with col2:
                    st.metric("Medium Confidence", confidence_counts.get("Medium", 0))
                with col3:
                    st.metric("Low Confidence", confidence_counts.get("Low", 0))
                with col4:
                    st.metric("Not Found", confidence_counts.get("Not Found", 0))

                st.subheader("Updated BOM with Verification")
                # Add verification results to BOM
                bom_mapped["VERIFIED"] = verification_df["Match Found"]
                bom_mapped["CONFIDENCE"] = verification_df["Confidence"]
                st.dataframe(bom_mapped)

                st.subheader("Final ERP Excel")
                st.dataframe(erp_data)

                updated_bom_excel = io.BytesIO()
                bom_mapped.to_excel(updated_bom_excel, index=False)
                updated_bom_excel.seek(0)

                erp_excel = io.BytesIO()
                erp_data.to_excel(erp_excel, index=False)
                erp_excel.seek(0)

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="Download Updated BOM",
                        data=updated_bom_excel,
                        file_name="verified_bom.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                with col2:
                    st.download_button(
                        label="Download ERP Excel",
                        data=erp_excel,
                        file_name="final_erp.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload the BOM Excel file to continue.")

        st.subheader("Expected BOM File Format")
        sample_bom = pd.DataFrame(
            {
                "SR. NO.": [1, 2, 3],
                "DESCRIPTION": [
                    "MCCB 3P 100A 36kA TM",
                    "Contactor 3P 95A 415V AC",
                    "LED Indicator Light 230V AC Green",
                ],
                "MAKE": ["Schneider", "ABB", "Siemens"],
                "TOTAL QTY 2 SETS": [2, 4, 8],
                "TYPE REFERENCE": ["Q1", "K1", "L1"],
                "ITEM CODE": ["MC-100-36", "CON-95-AC", "IND-LED-GRN"],
            }
        )
        st.dataframe(sample_bom)


if __name__ == "__main__":
    main()
