import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import NearestNeighbors

# Set page configuration
st.set_page_config(page_title="Anti-Plasmodial Activity Predictor", layout="wide")

# ==============================================================================
# MOLECULAR FEATURIZATION PIPELINE
# ==============================================================================
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    """Converts a SMILES string into a 2048-bit binary ECFP4 fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVinter(mol, radius, nBits=nBits)
        # Convert explicit RDKit vector into a standard NumPy array of 0s and 1s
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

# ==============================================================================
# MODEL & APD THRESHOLD LOADING ENGINE
# ==============================================================================
@st.cache_resource
def load_model_artifacts():
    """
    Downloads model assets dynamically, loads them, and calculates a 
    rigorous, mathematically corrected APD threshold baseline.
    """
    MODEL_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/malaria_rf_ecfp4_model.joblib"
    FEATURES_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/ecfp4_features.npy"
    
    model_path = "malaria_rf_ecfp4_model.joblib"
    features_path = "ecfp4_features.npy"
    
    def download_large_file(url, destination):
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            block_size = 1024 * 1024  
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)

    status_box = st.empty()
    
    if not os.path.exists(model_path):
        status_box.info("📥 Downloading core Random Forest model from Release assets...")
        download_large_file(MODEL_URL, model_path)
        
    if not os.path.exists(features_path):
        status_box.info("📥 Downloading ECFP4 training feature matrix...")
        download_large_file(FEATURES_URL, features_path)
        
    status_box.success("🎉 All machine learning core assets loaded successfully!")
            
    # Load model and features into server memory
    model = joblib.load(model_path)
    X_train = np.load(features_path) 
    
    # 🛠️ FIXED APD LOGIC: Request 6 neighbors instead of 5
    # The absolute closest neighbor to a training sample evaluated against itself is always 0.0.
    # We pull 6 neighbors so we can discard that artificial 0.0 self-match column.
    nn = NearestNeighbors(n_neighbors=6, metric='jaccard', n_jobs=-1)
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    
    # Slice out the first column [:, 1:] to remove the 0.0 self-matches. 
    # This leaves us with the true 5 closest neighbor distances for every compound.
    real_neighbors_distances = distances[:, 1:]
    mean_distances = np.mean(real_neighbors_distances, axis=1)
    
    # Compute the fair, accurate 95th percentile threshold
    apd_threshold = np.percentile(mean_distances, 95)
    
    status_box.empty()
    return model, nn, apd_threshold

# Safe execution sequence to initiate the backend
try:
    model, nn_engine, APD_THRESHOLD_CONSTANT = load_model_artifacts()
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
* **Calculated Training APD Threshold Limit:** `{APD_THRESHOLD_CONSTANT:.4f}`
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
        st.error("❌ Invalid SMILES structure string. Please verify chemical notation and case-sensitivity.")
    else:
        X_single = np.array([fp])
        
        pred = model.predict(X_single)[0]
        prob = model.predict_proba(X_single)[0][1]
        
        # Pull 5 nearest neighbors (no self-matches exist for a fresh, unseen molecule input)
        dist, _ = nn_engine.kneighbors(X_single, n_neighbors=5)
        mean_dist =
