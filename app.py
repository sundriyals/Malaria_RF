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
    # URLs pointing directly to your new GitHub Release assets
    MODEL_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/malaria_rf_ecfp4_model.joblib"
    FEATURES_URL = "https://github.com/sundriyals/Malaria_RF/releases/download/v1.0.0/ecfp4_features.npy"
    
    # Local destination names
    model_path = "malaria_rf_ecfp4_model.joblib"
    features_path = "ecfp4_features.npy"
    
    # Download assets if they don't exist in the current working directory
    with st.spinner("Downloading machine learning core assets from repository release..."):
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(MODEL_URL, model_path)
        if not os.path.exists(features_path):
            urllib.request.urlretrieve(FEATURES_URL, features_path)
            
    # Load assets into memory
    model = joblib.load(model_path)
    X_train = np.load(features_path) 
    
    # Compute APD Threshold dynamically
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

uploaded_file = st.file_uploader("Choose a CSV file to screen", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    smiles_col = [col for col in df_input.columns if col.lower() in ['smiles', 'smiles string', 'structure']]
    
    if not smiles_col:
        st.error("❌ Column Error: Could not find a 'Smiles' column in your CSV. Please check headers.")
    else:
        target_col = smiles_col[0]
        st.success(f"Processing data using column: '{target_col}'")
        
        features = []
        valid_indices = []
        
        with st.spinner("Featurizing structures and parsing chemical space boundaries..."):
            for idx, smiles in enumerate(df_input[target_col]):
                fp = smiles_to_ecfp4(str(smiles).strip())
                if fp is not None:
                    features.append(fp)
                    valid_indices.append(idx)
                    
        if len(features) == 0:
            st.error("❌ Format Error: No valid SMILES strings could be processed.")
        else:
            X_screen = np.array(features)
            
            # Column 1 predictions
            preds = model.predict(X_screen)
            probs = model.predict_proba(X_screen)[:, 1]
            
            activity_column = []
            for pred, prob in zip(preds, probs):
                class_label = "🟢 Active" if pred == 1 else "🔴 Inactive"
                activity_column.append(f"{class_label} ({prob*100:.1f}%)")
                
            # Column 2 APD checks
            distances, _ = nn_engine.kneighbors(X_screen)
            mean_distances = np.mean(distances, axis=1)
            
            apd_column = []
            for dist in mean_distances:
                if dist <= APD_THRESHOLD:
                    apd_column.append("✅ Reliable (Inside APD)")
                else:
                    apd_column.append("⚠️ Unreliable (Extrapolation / Outside APD)")
            
            results_df = pd.DataFrame({
                "SMILES": df_input[target_col],
                "Activity": [activity_column[valid_indices.index(i)] if i in valid_indices else "❌ Invalid SMILES" for i in range(len(df_input))],
                "APD": [apd_column[valid_indices.index(i)] if i in valid_indices else "N/A" for i in range(len(df_input))]
            })
            
            st.write("### 📊 Screening Preview Results")
            st.dataframe(results_df, use_container_width=True)
            
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Screened Compounds Table",
                data=csv_export,
                file_name="malaria_screening_results.csv",
                mime="text/csv"
            )
