import streamlit as st
from src.data_loader import load_multiple_csvs

def load_data_section():
    st.header("📂 Load Data")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        combined_df, separate_dfs = load_multiple_csvs(uploaded_files)

        # Store combined in session state
        if not combined_df.empty:
            st.session_state.raw_data = combined_df
            st.session_state.current_data = combined_df.copy()
            st.success(f"Loaded and combined {len(uploaded_files)} file(s) using Parquet optimization. Shape: {combined_df.shape}")
            st.dataframe(combined_df.head(20))

        # Show separately stored mismatched schema files
        if separate_dfs:
            st.warning("Some files had mismatched schemas and were not concatenated.")
            for fname, df in separate_dfs.items():
                st.subheader(f"Data preview: {fname}")
                st.dataframe(df.head(10))

    else:
        st.info("Please upload one or more CSV files to begin.")