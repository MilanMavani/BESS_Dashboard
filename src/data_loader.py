import pandas as pd
import streamlit as st
import io
import os
import pyarrow 

# --- NEW: Optimization Function ---
def optimize_dtypes(df):
    """
    Optimizes data types to reduce memory usage.
    Converts object columns to categories if they have low cardinality (few unique values).
    """
    for col in df.columns:
        # Only process object (string) columns
        if df[col].dtype == 'object':
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            
            # If less than 50% of values are unique, it's efficient to use category
            # (e.g., Device IDs, Status Labels, Error Codes)
            if num_total_values > 0 and (num_unique_values / num_total_values) < 0.5:
                df[col] = df[col].astype('category')
    return df
# ----------------------------------

def load_data(file, file_path, separator, bad_lines_action, file_type, skiprows=None):
    try:
        if file_type == 'csv':
            st.info(f"Attempting to load CSV with delimiter: '{separator}', skipping rows: {skiprows if skiprows is not None else 'None'}")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                df = pd.read_csv(
                    f,
                    sep=separator,
                    on_bad_lines=bad_lines_action, 
                    engine='python',
                    skiprows=skiprows 
                )
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            st.error("Unsupported file type selected.")
            return None
        
        if df.empty or len(df.columns) <= 1:
            if len(df.columns) == 1 and file_type == 'csv':
                st.error(f"Data parsed into a single column. The separator '{separator}' is likely incorrect.")
            else:
                st.error("The loaded data is either empty or could not be parsed correctly.")
            return None

        # --- NEW: Apply Optimization here ---
        df = optimize_dtypes(df)
        # ------------------------------------

        return df

    except Exception as e:
        st.error(f"An error occurred while loading the data. Error: {e}")
        return None



def convert_to_parquet(uploaded_file, separator, bad_lines_action, skiprows=None):

    try:
        # Read CSV directly from the uploaded file buffer
        uploaded_file.seek(0)
        csv_df = pd.read_csv(
            uploaded_file, 
            sep=separator, 
            on_bad_lines=bad_lines_action, 
            engine='python',
            skiprows=skiprows # This is where the argument is used
        )
        
        if csv_df.empty:
            st.warning("The CSV file is empty. Cannot convert to Parquet.")
            return None, None
            
        buffer = io.BytesIO()
        csv_df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        # Determine the new filename
        file_name, _ = os.path.splitext(uploaded_file.name)
        new_file_name = f"{file_name}.parquet"
        
        return buffer.getvalue(), new_file_name

    except Exception as e:
        st.error(f"Failed to convert file to Parquet. Check the CSV profile settings. Error: {e}")
        return None, None