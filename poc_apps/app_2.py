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


def find_similar_item(row, master_catalog, preprocessed_catalog):
    if not isinstance(row["DESCRIPTION"], str) or pd.isna(row["DESCRIPTION"]):
        return None

    description = row["DESCRIPTION"]
    family = row["FAMILY"]
    category = row["CATEGORY"]
    ac_dc = row["AC/DC"]
    make = row["MAKE"] if "MAKE" in row and isinstance(row["MAKE"], str) else ""

    filtered_catalog = master_catalog
    if family and family != "OTHER":
        filtered_catalog = filtered_catalog[
            filtered_catalog["Family"].str.upper() == family.upper()
        ]

    if len(filtered_catalog) == 0:
        filtered_catalog = master_catalog

    if category and category != "OTHER":
        filtered_catalog = filtered_catalog[
            filtered_catalog["Category"].str.upper() == category.upper()
        ]

    if len(filtered_catalog) == 0:
        filtered_catalog = master_catalog

    if make and make.strip():
        make_filtered = filtered_catalog[
            filtered_catalog["Make"].str.upper() == make.upper()
        ]
        if len(make_filtered) > 0:
            filtered_catalog = make_filtered

    if len(filtered_catalog) == 0:
        return None

    vectorizer = TfidfVectorizer(stop_words="english")

    filtered_indices = filtered_catalog.index.tolist()

    filtered_descriptions = [preprocessed_catalog[i] for i in filtered_indices]

    try:
        tfidf_matrix = vectorizer.fit_transform(filtered_descriptions + [description])

        cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

        if len(cosine_similarities) > 0 and max(cosine_similarities) > 0.3:
            most_similar_idx = filtered_indices[np.argmax(cosine_similarities)]
            return master_catalog.loc[most_similar_idx]
    except:
        pass

    return None


def preprocess_description(description):
    if not isinstance(description, str):
        return ""

    text = description.upper()

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    st.title("Technoocrafts BOM to ERP Format Converter")

    with st.sidebar:
        st.header("Upload Files")
        uploaded_bom = st.file_uploader("Upload BOM Excel File", type=["xlsx", "xls"])

    if uploaded_bom:
        try:
            bom_data = pd.read_excel(uploaded_bom)

            st.subheader("BOM File Preview")
            st.dataframe(bom_data.head())

            st.subheader("Master Catalog Preview")

            st.subheader("Column Mapping")

            col1, col2, col3, col4 = st.columns(4)

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

                with st.spinner("Extracting features from descriptions..."):
                    bom_mapped["FAMILY"] = bom_mapped["DESCRIPTION"].apply(
                        extract_family
                    )
                    bom_mapped["CATEGORY"] = bom_mapped["DESCRIPTION"].apply(
                        extract_category
                    )
                    bom_mapped["AC/DC"] = bom_mapped["DESCRIPTION"].apply(extract_ac_dc)

                preprocessed_catalog = [
                    preprocess_description(desc)
                    for desc in master_catalog["Item Description"]
                ]

                with st.spinner("Matching items with master catalog..."):
                    matched_items = []
                    for idx, row in bom_mapped.iterrows():
                        matched_item = find_similar_item(
                            row, master_catalog, preprocessed_catalog
                        )
                        if matched_item is not None:
                            matched_items.append(
                                {
                                    "Index": idx,
                                    "Item Code": matched_item["Item Code"],
                                    "Match": matched_item["Item Description"],
                                    "Confidence": (
                                        "High"
                                        if matched_item["Family"].upper()
                                        == row["FAMILY"].upper()
                                        else "Medium"
                                    ),
                                }
                            )
                        else:
                            matched_items.append(
                                {
                                    "Index": idx,
                                    "Item Code": "NOT_FOUND",
                                    "Match": "",
                                    "Confidence": "Low",
                                }
                            )

                matching_results = pd.DataFrame(matched_items)

                bom_mapped["ITEM_CODE"] = "NOT_FOUND"
                for _, row in matching_results.iterrows():
                    if row["Item Code"] != "NOT_FOUND":
                        bom_mapped.at[row["Index"], "ITEM_CODE"] = row["Item Code"]

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

                st.write("Matching Results with Master Catalog")
                st.dataframe(matching_results)

                st.subheader("Updated BOM with Item Codes")
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

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload both the BOM Excel file Excel file to continue.")

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
            }
        )
        st.dataframe(sample_bom)


if __name__ == "__main__":
    main()
