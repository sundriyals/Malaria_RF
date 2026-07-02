import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request
import urllib.error
import warnings
import traceback

# Silence scikit-learn's Jaccard boolean data conversion warnings safely
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set page configuration
st.set_page_config(page_title="Anti-Plasmodial Activity Predictor", layout="wide")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
APD_THRESHOLD_CONSTANT = 0.6744

# ==============================================================================
# MOLECULAR FEATURIZATION PIPELINE
# ==============================================================================
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    """Converts a SMILES string into a 2048-bit binary ECFP4 fingerprint."""
    try:
        smiles = str(smiles).strip()
        if not smiles or smiles == "nan" or smiles.lower() == "none":
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        
        # Fallback for complex structures
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
                
        if mol is None:
            return None
            
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

# ==============================================================================
# MODEL & APD LOADING ENGINE
# ==============================================================================
@st.cache_resource
def load_model_artifacts():
    """Downloads core model assets dynamically with explicit error tracing."""
    MODEL_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/malaria_rf_ecfp4_model.joblib"
    FEATURES_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/ecfp4_features.npy"
    
    model_path = "malaria_rf_ecfp4_model.joblib"
    features_path = "ecfp4_features.npy"
    
    def download_large_file(url, destination):
        try:
            with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
                block_size = 1024 * 1024  
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP {e.code} Error: Could not download asset from URL: {url}.")

    status_box = st.empty()
    
    if not os.path.exists(model_path):
        status_box.info("📥 Downloading core Random Forest model...")
        download_large_file(MODEL_URL, model_path)
        
    if not os.path.exists(features_path):
        status_box.info("📥 Downloading ECFP4 training feature matrix...")
        download_large_file(FEATURES_URL, features_path)
        
    status_box.success("🎉 Web assets initialized cleanly!")
            
    model = joblib.load(model_path)
    X_train = np.load(features_path) 
    
    nn = NearestNeighbors(n_neighbors=5, metric='jaccard', n_jobs=-1)
    nn.fit(X_train)
    
    status_box.empty()
    return model, nn

try:
    # Importing RDKit elements just in case it wasn't pre-initialized globally
    from rdkit import Chem
    from rdkit.Chem import AllChem
    model, nn_engine = load_model_artifacts()
except Exception as e:
    st.error(f"⚠️ App Core Initialization Error: {e}")
    st.stop()

# ==============================================================================
# USER INTERFACE LAYOUT
# ==============================================================================
st.title("🔬 Anti-Plasmodial Activity & APD Screening Portal")
st.markdown(f"""
Upload screening candidates containing structural **SMILES** strings to generate machine learning predictions. 
All calculations are backed by an automated **Applicability Domain (APD)** validation metric.
* **Active APD Threshold Limit:** `{APD_THRESHOLD_CONSTANT:.4f}`
""")

# ==============================================================================
# SINGLE COMPOUND QUICK SCREEN
# ==============================================================================
st.write("### 🧪 Single Compound Quick Screen")
single_smiles = st.text_input("Paste a single SMILES string here (e.g., CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO):")

if single_smiles:
    single_smiles = single_smiles.strip()
    fp = smiles_to_ecfp4(single_smiles)
    
    if fp is None:
        st.error("❌ Invalid SMILES structure string. Please verify chemical notation.")
    else:
        X_single = np.array([fp])
        
        pred = model.predict(X_single)[0]
        prob = model.predict_proba(X_single)[0][1]
        
        dist, _ = nn_engine.kneighbors(X_single, n_neighbors=5)
        mean_dist = np.mean(dist)
        
        activity_res = "Active" if pred == 1 else "Inactive"
        apd_res = "Reliable" if mean_dist <= APD_THRESHOLD_CONSTANT else "Unreliable"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Predicted Class", value=activity_res)
        with col2:
            st.metric(label="Probability Score", value=f"{prob*100:.1f}%")
        with col3:
            st.metric(label="Calculated Distance", value=f"{mean_dist:.4f}")
        with col4:
            st.metric(label="APD Status", value=apd_res)

st.markdown("---")

# ==============================================================================
# BATCH FILE HIGH-THROUGHPUT PROCESSING PORTAL
# ==============================================================================
st.write("### 📂 Batch File High-Throughput Screening")
uploaded_file = st.file_uploader("Choose a CSV file to screen", type=["csv"])

if uploaded_file is not None:
    try:
        header_df = pd.read_csv(uploaded_file, nrows=2)
        uploaded_file.seek(0) 
    except Exception as e:
        st.error(f"❌ Read Error: Could not parse CSV document structure. {e}")
        st.stop()

    smiles_col = [col for col in header_df.columns if col.lower() in ['smiles', 'smiles string', 'structure']]
    
    if not smiles_col:
        st.error("❌ Column Error: Could not find a 'Smiles' or 'Structure' column header in your CSV file.")
    else:
        target_col = smiles_col[0]
        st.success(f"Processing structural pipeline using column: '{target_col}'")
        
        processed_chunks = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_rows = 0
        
        try:
            with st.spinner("Streaming chemical spaces and processing APD constraints..."):
                for chunk in pd.read_csv(uploaded_file, chunksize=5000):
                    chunk = chunk.reset_index(drop=True)
                    
                    # Compute fingerprints rapidly using pandas .apply() instead of an explicit row loop
                    fingerprints = chunk[target_col].apply(smiles_to_ecfp4)
                    valid_mask = fingerprints.notna()
                    
                    # Create default placeholder columns preserving original user data
                    chunk["Activity Prediction"] = "Invalid SMILES structure"
                    chunk["Probability Score"] = "N/A"
                    chunk["Mean Neighbor Distance"] = "N/A"
                    chunk["APD Status"] = "N/A"
                    
                    if valid_mask.any():
                        # Extract only valid fingerprints to run ML on
                        X_screen = np.array(list(fingerprints[valid_mask]), dtype=np.int8)
                        
                        # Generate batch predictions
                        preds = model.predict(X_screen)
                        probs = model.predict_proba(X_screen)[:, 1]
                        
                        distances, _ = nn_engine.kneighbors(X_screen, n_neighbors=5)
                        mean_distances = np.mean(distances, axis=1)
                        
                        # Vectorized formatting assignments back into the dataframe structure
                        chunk.loc[valid_mask, "Activity Prediction"] = ["Active" if p == 1 else "Inactive" for p in preds]
                        chunk.loc[valid_mask, "Probability Score"] = [f"{prob*100:.1f}%" for prob in probs]
                        chunk.loc[valid_mask, "Mean Neighbor Distance"] = [f"{d:.4f}" for d in mean_distances]
                        chunk.loc[valid_mask, "APD Status"] = ["Reliable" if d <= APD_THRESHOLD_CONSTANT else "Unreliable" for d in mean_distances]
                    
                    processed_chunks.append(chunk)
                    processed_rows += len(chunk)
                    status_text.text(f"Processed structural records: {processed_rows:,}")

            progress_bar.progress(100)
            status_text.text(f"Complete! Total evaluated compounds: {processed_rows:,}")
            
            # Combine the processed chunks smoothly
            results_df = pd.concat(processed_chunks, ignore_index=True)
            
            st.write("### 📊 Screening Preview Results (First 500 Records)")
            st.dataframe(results_df.head(500), use_container_width=True)
            
            # Memory-efficient download extraction
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Screened Compounds Table",
                data=csv_export,
                file_name="malaria_screening_results.csv",
                mime="text/csv"
            )
            
        except Exception as batch_error:
            st.error("❌ An unexpected pipeline tracking error occurred during processing.")
            st.code(traceback.format_exc(), language="python")
