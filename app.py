# C:\Projects\invoice_extraction_project\app.py

import streamlit as st
import os
import google.generativeai as genai
import pdfplumber
import pandas as pd
import json
from datetime import datetime
from io import BytesIO # Required for in-memory Excel file

# --- Helper functions (Copied and adapted from our previous discussions) ---

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

def extract_text_from_pdf(uploaded_file):
    """
    Extracts all text from an uploaded Streamlit PDF file using pdfplumber.
    Returns the extracted text as a string, or None if an error occurs.
    """
    text = ""
    try:
        # pdfplumber.open can accept a file-like object directly
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\\n"
        return text
    except Exception as e:
        st.error(f"❌ Error extracting text from {uploaded_file.name}: {e}")
        st.info("Ensure the PDF is not corrupted or password-protected.")
        return None

def get_gemini_model():
    """
    Attempts to find and return a Gemini model that supports 'generateContent'.
    Prioritizes 'gemini-1.5-flash'.
    Returns a GenerativeModel instance or None if no compatible model is found.
    """
    try:
        # Try to use gemini-1.5-flash first
        for m in genai.list_models():
            if m.name == "models/gemini-1.5-flash" and 'generateContent' in m.supported_generation_methods:
                st.info("Using model: models/gemini-1.5-flash")
                return genai.GenerativeModel(m.name)
        
        # Fallback to any other model that supports generateContent
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

def parse_invoice_with_gemini(model, invoice_text, file_name=""):
    """
    Sends invoice text to the Gemini model with a prompt to extract specific fields.
    Returns the parsed data as a dictionary, or None if parsing fails.
    Includes explicit instructions for Buyer and Seller GSTIN.
    """
    if not invoice_text.strip():
        st.warning(f"⚠️ Invoice text for {file_name if file_name else 'a file'} is empty, skipping Gemini processing.")
        return None

    # Refined prompt to distinguish Buyer and Seller GSTIN
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
        
        # Attempt to clean and parse Gemini's response
        raw_json_str = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_json_str)

        # Basic type conversion/cleaning for numeric fields if needed
        # This helps ensure consistency in your Excel sheet
        numeric_fields = ["Taxable Value", "Invoice Value", "CGST", "SGST", "IGST"]
        for field in numeric_fields:
            if field in data and isinstance(data[field], str):
                try:
                    # Remove currency symbols, commas, and convert to float
                    data[field] = float(data[field].replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip())
                except ValueError:
                    data[field] = 0.0 # Default to 0 if conversion fails

        return data
    except json.JSONDecodeError as je:
        st.error(f"❌ Failed to parse Gemini's JSON response for {file_name}: {je}")
        st.code(f"Raw Response from Gemini (could not parse):\\n{response.text}" if response else "No response received.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during Gemini processing for {file_name}: {e}")
        st.code(f"Raw Response from Gemini (if available):\\n{response.text}" if response else "No response received.")
        return None


# --- Streamlit App Layout ---
st.set_page_config(page_title="Gemini Invoice Parser", layout="centered")

st.title("📄 Invoice Extraction Tool using Google Gemini AI")

st.markdown("""
Welcome to the Invoice Extraction Tool!
Upload your PDF invoices and let Gemini AI extract key details for you.
""")

# Input for Gemini API Key
api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password", help="Your API key will not be stored.")

# File Uploader for PDF
uploaded_files = st.file_uploader("📁 Upload PDF Invoice(s):", type="pdf", accept_multiple_files=True)

# A button to trigger the processing
if st.button("🚀 Process Invoices"):
    if not api_key:
        st.error("❌ Please enter your Gemini API Key to proceed.")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file to process.")
    else:
        st.info("Starting invoice processing... This might take a moment depending on file size and number of invoices.")

        # 1. Configure Gemini API
        with st.spinner("Configuring Gemini API..."):
            if not configure_gemini(api_key):
                st.stop() # Stop execution if API setup fails

        # 2. Get a suitable Gemini model
        with st.spinner("Finding a suitable Gemini model..."):
            model = get_gemini_model()
            if not model:
                st.stop() # Stop execution if no model is found

        all_invoices_data = [] # List to store extracted data from all invoices

        # 3. Process each uploaded file
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            
            # Use Streamlit's BytesIO for in-memory processing, no need for temp files on disk
            # Reset file pointer for pdfplumber if it was read previously
            uploaded_file.seek(0) 

            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                invoice_text = extract_text_from_pdf(uploaded_file)
                if not invoice_text:
                    continue # Skip to next file if text extraction fails

            with st.spinner(f"Sending text from {uploaded_file.name} to Gemini for parsing..."):
                parsed_data = parse_invoice_with_gemini(model, invoice_text, uploaded_file.name)
                
                if parsed_data:
                    # Add original filename for traceability in the Excel sheet
                    parsed_data['Source_File'] = uploaded_file.name
                    all_invoices_data.append(parsed_data)
                    st.success(f"✅ Successfully extracted data from {uploaded_file.name}.")
                    st.json(parsed_data) # Display raw JSON for verification
                else:
                    st.error(f"❌ Failed to extract structured data from {uploaded_file.name}. Check logs above.")

        # 4. Save all extracted data to Excel and provide download link
        if all_invoices_data:
            st.success("🎉 All available invoices processed! Consolidating data...")
            try:
                df = pd.DataFrame(all_invoices_data)
                
                # Reorder columns to put Source_File at the beginning for better readability
                cols = ['Source_File'] + [col for col in df.columns if col != 'Source_File']
                df = df[cols]

                st.dataframe(df)

                # Create an Excel file in memory
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
                st.success("Your Excel file is ready for download!")

            except Exception as e:
                st.error(f"❌ Error preparing or saving data to Excel: {e}")
                st.info("Please ensure 'openpyxl' is installed (pip install openpyxl).")
        else:
            st.warning("No data could be successfully extracted from any of the uploaded PDFs.")

st.markdown("---")
st.write("Developed with using Streamlit and Gemini AI.")