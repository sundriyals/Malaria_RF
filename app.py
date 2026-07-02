import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request
import urllib.error
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors

# Set page configuration
st.set_page_config(page_title="Anti-Plasmodial Activity Predictor", layout="wide")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# Your exact calculated threshold value from Google Colab
APD_THRESHOLD_CONSTANT = 0.6744

# ==============================================================================
# MOLECULAR FEATURIZATION PIPELINE
# ==============================================================================
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    """Converts a SMILES string into a 2048-bit binary ECFP4 fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

# ==============================================================================
# MODEL & APD LOADING ENGINE (WITH DETAILED ERROR DEBUGGING)
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
            # This will pinpoint exactly which URL is broken if a 404 occurs
            raise Exception(
                f"HTTP {e.code} Error: Could not download asset from URL: {url}. "
                f"Please verify that your GitHub Release tag is exactly 'v1.0.0' and the file asset name matches perfectly."
            )

    status_box = st.empty()
    
    if not os.path.exists(model_path):
        status_box.info("📥 Downloading core Random Forest model...")
        download_large_file(MODEL_URL, model_path)
        
    if not os.path.exists(features_path):
        status_box.info("📥 Downloading ECFP4 training feature matrix...")
        download_large_file(FEATURES_URL, features_path)
        
    status_box.success("🎉 Web assets initialized cleanly!")
            
    # Load components into server memory
    model = joblib.load(model_path)
    X_train = np.load(features_path) 
    
    # Fit the engine using our training array
    nn = NearestNeighbors(n_neighbors=5, metric='jaccard', n_jobs=-1)
    nn.fit(X_train)
    
    status_box.empty()
    return model, nn

# Safe execution sequence
try:
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
        
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_rows = 0
        
        with st.spinner("Streaming chemical spaces and processing APD constraints..."):
            for chunk in pd.read_csv(uploaded_file, chunksize=5000):
                chunk = chunk.reset_index(drop=True)
                chunk_features = []
                chunk_valid_indices = []
                
                for idx, smiles in enumerate(chunk[target_col]):
                    fp = smiles_to_ecfp4(str(smiles).strip())
                    if fp is not None:
                        chunk_features.append(fp)
                        chunk_valid_indices.append(idx)
                
                if len(chunk_features) > 0:
                    X_screen = np.array(chunk_features)
                    
                    preds = model.predict(X_screen)
                    probs = model.predict_proba(X_screen)[:, 1]
                    
                    distances, _ = nn_engine.kneighbors(X_screen, n_neighbors=5)
                    mean_distances = np.mean(distances, axis=1)
                    
                    activity_labels = ["Active" if p == 1 else "Inactive" for p in preds]
                    probability_scores = [f"{prob*100:.1f}%" for prob in probs]
                    apd_labels = ["Reliable" if d <= APD_THRESHOLD_CONSTANT else "Unreliable" for d in mean_distances]
                    
                    for idx in range(len(chunk)):
                        # Retain all original columns from user uploaded CSV
                        row_dict = chunk.iloc[idx].to_dict()
                        
                        if idx in chunk_valid_indices:
                            list_pos = chunk_valid_indices.index(idx)
                            row_dict["Activity Prediction"] = activity_labels[list_pos]
                            row_dict["Probability Score"] = probability_scores[list_pos]
                            row_dict["Mean Neighbor Distance"] = f"{mean_distances[list_pos]:.4f}"
                            row_dict["APD Status"] = apd_labels[list_pos]
                        else:
                            row_dict["Activity Prediction"] = "Invalid SMILES structure"
                            row_dict["Probability Score"] = "N/A"
                            row_dict["Mean Neighbor Distance"] = "N/A"
                            row_dict["APD Status"] = "N/A"
                        all_results.append(row_dict)
                else:
                    for idx in range(len(chunk)):
                        row_dict = chunk.iloc[idx].to_dict()
                        row_dict["Activity Prediction"] = "Invalid SMILES structure"
                        row_dict["Probability Score"] = "N/A"
                        row_dict["Mean Neighbor Distance"] = "N/A"
                        row_dict["APD Status"] = "N/A"
                        all_results.append(row_dict)
                
                processed_rows += len(chunk)
                status_text.text(f"Processed structural records: {processed_rows:,}")

        progress_bar.progress(100)
        status_text.text(f"Complete! Total evaluated compounds: {processed_rows:,}")
        
        results_df = pd.DataFrame(all_results)
        st.write("### 📊 Screening Preview Results (First 500 Records)")
        st.dataframe(results_df.head(500), use_container_width=True)
        
        csv_export = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Screened Compounds Table",
            data=csv_export,
            file_name="malaria_screening_results.csv",
            mime="text/csv"
        )
