# C:\Projects\invoice_extraction_project\app.py

import streamlit as st
import os
import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
import re
from datetime import datetime
from io import BytesIO

# --- Helper functions ---

def configure_gemini(api_key):
    """
    Configures the Google Gemini API with the provided API key.
    Returns True on success, False on failure.
    """
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"❌ Gemini API configuration failed: {e}")
        st.info("Please ensure your API key is correct and valid. You can get one from https://aistudio.google.com/app/apikey")
        return False

# MODIFIED: Now returns a list of texts, one for each page
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from each page of an uploaded Streamlit PDF file using pdfplumber.
    Returns a list of strings, where each string is the text from one page.
    Returns an empty list if an error occurs or no text is found.
    """
    page_texts = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
                else:
                    st.warning(f"⚠️ No text found on page {i+1} of {uploaded_file.name}. Skipping this page.")
        return page_texts
    except Exception as e:
        st.error(f"❌ Error extracting text from {uploaded_file.name}: {e}")
        st.info("Ensure the PDF is not corrupted or password-protected.")
        return [] # Return empty list on error

def get_gemini_model():
    """
    Attempts to find and return a Gemini model that supports 'generateContent'.
    Prioritizes 'models/gemini-1.5-flash'.
    Returns a GenerativeModel instance or None if no compatible model is found.
    """
    try:
        for m in genai.list_models():
            if m.name == "models/gemini-1.5-flash" and 'generateContent' in m.supported_generation_methods:
                st.info("Using model: models/gemini-1.5-flash")
                return genai.GenerativeModel(m.name)
        
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                st.info(f"Using fallback model: {m.name}")
                return genai.GenerativeModel(m.name)
        
        st.error("❌ No compatible Gemini model found that supports 'generateContent'.")
        st.info("Please check your API key and available models in Google AI Studio. Your quota might also be exhausted.")
        return None
    except Exception as e:
        st.error(f"❌ Error listing Gemini models: {e}")
        st.info("Ensure you have a stable internet connection and a valid API key.")
        return None

def parse_invoice_with_gemini(model, invoice_text, file_name="", page_number=None):
    """
    Sends invoice text to the Gemini model with a prompt to extract specific fields.
    Returns the parsed data as a dictionary, or None if parsing fails.
    Includes explicit instructions for Buyer and Seller GSTIN.
    """
    invoice_id = f" (Page {page_number})" if page_number is not None else ""
    
    if not invoice_text.strip():
        st.warning(f"⚠️ Invoice text for {file_name}{invoice_id} is empty, skipping Gemini processing.")
        return None

    prompt = """
You are an intelligent invoice parser. Your task is to extract the following information from the given invoice text:

- Invoice Number
- Invoice Date (format as YYYY-MM-DD if possible)
- Party Name (This refers to the Buyer/Recipient's full company or individual name)
- Buyer GSTIN (This refers to the Goods and Services Tax Identification Number of the Buyer/Recipient)
- Seller GSTIN (This refers to the Goods and Services Tax Identification Number of the Seller/Supplier)
- Taxable Value
- Invoice Value
- CGST (Central Goods and Services Tax amount)
- SGST (State Goods and Services Tax amount)
- IGST (Integrated Goods and Services Tax amount)

Return the result as a JSON object with these exact field names.
If a field is not found or is not applicable, use "N/A" for string values (like GSTIN or Party Name) or "0" for numeric values (like Taxable Value, CGST, SGST, IGST).
Do not include any extra explanation, markdown, or other text outside the JSON object.

Here is the invoice text:
"""
    full_prompt = prompt + invoice_text

    response = None 
    try:
        response = model.generate_content(full_prompt)
        
        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_json_str)

        numeric_fields = ["Taxable Value", "Invoice Value", "CGST", "SGST", "IGST"]
        for field in numeric_fields:
            if field in data and isinstance(data[field], str):
                try:
                    clean_value = data[field].replace('$', '').replace('€', '').replace('£', '').replace('₹', '').replace(',', '').strip()
                    data[field] = float(clean_value)
                except ValueError:
                    data[field] = 0.0

        return data
    except json.JSONDecodeError as je:
        st.error(f"❌ Failed to parse Gemini's JSON response for {file_name}{invoice_id}: {je}")
        st.code(f"Raw Response from Gemini (could not parse):\\n{response.text}" if response else "No response received.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during Gemini processing for {file_name}{invoice_id}: {e}")
        st.code(f"Raw Response from Gemini (if available):\\n{response.text}" if response else "No response received.")
        return None

# --- Removed the split_invoices_text function as we are now processing page by page ---

# --- Streamlit App Layout ---
st.set_page_config(page_title="Gemini Invoice Parser", layout="centered")

st.title("📄 Invoice Extraction Tool using Google Gemini AI")

st.markdown("""
Welcome to the Invoice Extraction Tool!
Upload your PDF invoices (whether single or merged) and let Gemini AI extract key details for you.
""")

# Input for Gemini API Key
api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password", help="Your API key will not be stored.")

# File Uploader for PDF
uploaded_files = st.file_uploader("📁 Upload PDF Invoice(s):", type="pdf", accept_multiple_files=True)

# A button to trigger the processing
if st.button("🚀 Process Invoices"):
    if not api_key:
        st.error("❌ Please enter your Gemini API Key to proceed.")
        st.stop()
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file to process.")
        st.stop()
    else:
        st.info("Starting invoice processing... This might take a moment depending on file size and number of invoices.")

        with st.spinner("Configuring Gemini API..."):
            if not configure_gemini(api_key):
                st.stop()

        with st.spinner("Finding a suitable Gemini model..."):
            model = get_gemini_model()
            if not model:
                st.stop()

        all_extracted_invoices_data = [] # List to store extracted data from ALL invoices

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files_to_process = len(uploaded_files)
        current_file_index = 0

        for uploaded_file in uploaded_files:
            current_file_index += 1
            progress_message = f"Processing file {current_file_index}/{total_files_to_process}: {uploaded_file.name}"
            status_text.text(progress_message)
            
            uploaded_file.seek(0) # Reset file pointer for pdfplumber if it was read previously

            # MODIFIED: Get text as a list of pages
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                page_texts = extract_text_from_pdf(uploaded_file)
                if not page_texts:
                    st.error(f"Failed to extract any text pages from {uploaded_file.name}. Skipping.")
                    progress_bar.progress(current_file_index / total_files_to_process)
                    continue
            
            st.info(f"Detected {len(page_texts)} page(s) in {uploaded_file.name}. Processing each page as a potential invoice.")

            # Iterate through each page's text
            for page_idx, page_text_content in enumerate(page_texts):
                invoice_identifier = f"{uploaded_file.name} (Page {page_idx+1})"
                
                with st.spinner(f"Sending page {page_idx+1} from {uploaded_file.name} to Gemini..."):
                    parsed_data = parse_invoice_with_gemini(model, page_text_content, invoice_identifier, page_idx + 1)
                    
                if parsed_data:
                    # Add original filename and page number for traceability
                    parsed_data['Source_File'] = uploaded_file.name
                    parsed_data['Original_File_Page'] = page_idx + 1 # Storing page number
                    all_extracted_invoices_data.append(parsed_data)
                    st.success(f"✅ Data extracted from {invoice_identifier}.")
                else:
                    st.error(f"❌ Failed to extract structured data from {invoice_identifier}. See errors above.")
            
            progress_bar.progress(current_file_index / total_files_to_process)

        status_text.text("Invoice processing complete!")
        progress_bar.empty()

        # 4. Save all extracted data to Excel and provide download link
        if all_extracted_invoices_data:
            st.success("🎉 All invoices processed! Your Excel file is ready.")
            try:
                df = pd.DataFrame(all_extracted_invoices_data)
                
                # Reorder columns to put Source_File and Original_File_Page at the beginning
                primary_cols = ['Source_File', 'Original_File_Page']
                cols = primary_cols + [col for col in df.columns if col not in primary_cols]
                df = df[cols]

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Extracted Invoices')
                processed_data = output.getvalue()

                st.download_button(
                    label="📥 Download Extracted Data as Excel",
                    data=processed_data,
                    file_name="extracted_invoices.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                
            except Exception as e:
                st.error(f"❌ Error preparing or saving data to Excel: {e}")
                st.info("Please ensure 'openpyxl' is installed (pip install openpyxl) and try again.")
        else:
            st.warning("No data could be successfully extracted from any of the uploaded PDFs.")

st.markdown("---")
st.write("Developed with using Streamlit and Gemini AI.")
