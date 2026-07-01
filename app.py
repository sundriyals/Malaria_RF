import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request
from rdkit import Chem
from rdkit.Chem import AllChem

st.set_page_config(page_title="Anti-Plasmodial Activity Predictor", layout="wide")

@st.cache_resource
def load_model_artifacts():
    """Downloads model files directly from GitHub Releases if not present locally, then loads them."""
    MODEL_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/malaria_rf_ecfp4_model.joblib"
    FEATURES_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/ecfp4_features.npy"
    
    model_path = "malaria_rf_ecfp4_model.joblib"
    features_path = "ecfp4_features.npy"
    
    with st.spinner("Downloading machine learning core assets from repository release..."):
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(MODEL_URL, model_path)
        if not os.path.exists(features_path):
            urllib.request.urlretrieve(FEATURES_URL, features_path)
            
    model = joblib.load(model_path)
    X_train = np.load(features_path) 
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5, metric='jaccard')
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    mean_distances = np.mean(distances, axis=1)
    apd_threshold = np.percentile(mean_distances, 95)
    
    return model, nn, apd_threshold

def smiles_to_ecfp4(smiles, n_bits=2048, radius=2):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None

# Execute load sequence
try:
    model, nn_engine, APD_THRESHOLD = load_model_artifacts()
except Exception as e:
    st.error(f"⚠️ Deployment Asset Error: {e}")
    st.stop()

# ==============================================================================
# USER INTERFACE (UI) LAYOUT
# ==============================================================================
st.title("🔬 Anti-Plasmodial Activity & APD Screening Portal")
st.markdown("""
Upload a screening `.csv` file containing structural **SMILES** strings to generate machine learning predictions 
backed by a rigorous **Applicability Domain (APD)** reliability metric.
""")

uploaded_file = st.file_uploader("Choose a CSV file to screen (Large files will be streamed)", type=["csv"])

if uploaded_file is not None:
    # Read just the header first to find the SMILES column
    try:
        header_df = pd.read_csv(uploaded_file, nrows=2)
        uploaded_file.seek(0) # Reset file pointer back to start
    except Exception as e:
        st.error(f"❌ Read Error: Could not read file structure. {e}")
        st.stop()

    smiles_col = [col for col in header_df.columns if col.lower() in ['smiles', 'smiles string', 'structure']]
    
    if not smiles_col:
        st.error("❌ Column Error: Could not find a 'Smiles' column in your CSV. Please check headers.")
    else:
        target_col = smiles_col[0]
        st.success(f"Processing data using column: '{target_col}'")
        
        # Placeholder for collected results
        all_results = []
        
        # UI controls for monitoring stream status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_rows_estimated = 100000 
        processed_rows = 0
        
        with st.spinner("Streaming chemical spaces and calculating APD constraints..."):
            # Process the CSV file iteratively in 5,000-row chunks to protect system RAM
            for chunk in pd.read_csv(uploaded_file, chunksize=5000):
                chunk_features = []
                chunk_valid_indices = []
                
                for idx, smiles in enumerate(chunk[target_col]):
                    fp = smiles_to_ecfp4(str(smiles).strip())
                    if fp is not None:
                        chunk_features.append(fp)
                        chunk_valid_indices.append(idx)
                
                if len(chunk_features) > 0:
                    X_screen = np.array(chunk_features)
                    
                    # Core Predictions
                    preds = model.predict(X_screen)
                    probs = model.predict_proba(X_screen)[:, 1]
                    
                    activity_column = []
                    for pred, prob in zip(preds, probs):
                        # Safe text tags that will never break or corrupt in Excel/CSV formats
                        class_label = "Active" if pred == 1 else "Inactive"
                        activity_column.append(f"{class_label} ({prob*100:.1f}%)")
                        
                    # Applicability Domain Metric
                    distances, _ = nn_engine.kneighbors(X_screen)
                    mean_distances = np.mean(distances, axis=1)
                    
                    apd_column = []
                    for dist in mean_distances:
                        if dist <= APD_THRESHOLD:
                            apd_column.append("Reliable (Inside APD)")
                        else:
                            apd_column.append("Unreliable (Outside APD)")
                    
                    # Map batch outputs back to original frame alignment
                    for idx in range(len(chunk)):
                        if idx in chunk_valid_indices:
                            list_pos = chunk_valid_indices.index(idx)
                            all_results.append({
                                "SMILES": chunk.iloc[idx][target_col],
                                "Activity": activity_column[list_pos],
                                "APD": apd_column[list_pos]
                            })
                        else:
                            all_results.append({
                                "SMILES": chunk.iloc[idx][target_col],
                                "Activity": "Invalid SMILES structure",
                                "APD": "N/A"
                            })
                else:
                    for idx in range(len(chunk)):
                        all_results.append({
                            "SMILES": chunk.iloc[idx][target_col],
                            "Activity": "Invalid SMILES structure",
                            "APD": "N/A"
                        })
                
                processed_rows += len(chunk)
                progress_percent = min(int((processed_rows / total_rows_estimated) * 100), 99)
                progress_bar.progress(progress_percent)
                status_text.text(f"Processed rows: {processed_rows:,}")

        progress_bar.progress(100)
        status_text.text(f"Complete! Total processed structures: {processed_rows:,}")
        
        results_df = pd.DataFrame(all_results)
        
        st.write("### 📊 Screening Preview Results (First 500 Rows)")
        st.dataframe(results_df.head(500), use_container_width=True)
        
        csv_export = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Screened Compounds Table",
            data=csv_export,
            file_name="malaria_screening_results.csv",
            mime="text/csv"
        )
