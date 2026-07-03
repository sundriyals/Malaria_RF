import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request
import urllib.error
import warnings
import traceback

# Core Cheminformatics & Machine Learning Imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from sklearn.neighbors import NearestNeighbors

# Silence scikit-learn's Jaccard boolean data conversion warnings safely
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set page configuration
st.set_page_config(page_title="Anti-Plasmodial Activity Predictor", layout="wide")

# ==============================================================================
# CUSTOM VISUAL STYLING (Making Tabs Look Like Buttons)
# ==============================================================================
st.markdown("""
    <style>
        /* Target the tab container bar */
        div[data-testid="stTabs"] button {
            background-color: #f0f2f6;
            color: #31333F;
            border: 1px solid #dcdfe6;
            border-radius: 6px;
            padding: 10px 20px;
            margin-right: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        
        /* Hover effect over the tab buttons */
        div[data-testid="stTabs"] button:hover {
            background-color: #e4e7ed;
            border-color: #c0c4cc;
            color: #ff4b4b;
        }
        
        /* Active / Selected Tab Button styling */
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #ff4b4b !important;
            color: white !important;
            border-color: #ff4b4b !important;
            box-shadow: 0px 4px 6px rgba(255, 75, 75, 0.2);
        }
        
        /* Remove Streamlit's default red underline indicator line */
        div[data-testid="stTabs"] [data-baseweb="tab-highlight-bar"] {
            background-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
APD_THRESHOLD_CONSTANT = 0.6744

# ==============================================================================
# MOLECULAR FEATURIZATION & ADME PIPELINE
# ==============================================================================
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    """Converts a SMILES string into a 2048-bit binary ECFP4 fingerprint."""
    try:
        smiles = str(smiles).strip()
        if not smiles or smiles == "nan" or smiles.lower() == "none":
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        
        # Fallback for complex structures (e.g., hypervalent sulfoxides)
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

def compute_adme_lipinski(smiles):
    """Computes quantitative Lipinski descriptors and assigns a Pass/Fail status."""
    try:
        smiles = str(smiles).strip()
        if not smiles or smiles == "nan" or smiles.lower() == "none":
            return [np.nan, np.nan, np.nan, np.nan, "Invalid Structure"]
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan, np.nan, np.nan, np.nan, "Invalid Structure"]
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        # Lipinski Rule Violations Calculation (Max 1 violation allowed to pass)
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        
        pass_fail = "Pass" if violations <= 1 else "Fail"
        return [round(mw, 2), round(logp, 2), hbd, hba, pass_fail]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan, "Calculation Error"]

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
    model, nn_engine = load_model_artifacts()
except Exception as e:
    st.error(f"⚠️ App Core Initialization Error: {e}")
    st.stop()

# ==============================================================================
# USER INTERFACE LAYOUT (TABBED DESIGN)
# ==============================================================================
st.title("🔬 Anti-Plasmodial Activity & APD Screening Portal")

# Split web portal into distinct interactive views (now styled as buttons)
tab_screen, tab_metrics = st.tabs(["🧪 Screening Portal", "📊 Model Validation & Metrics"])

# ------------------------------------------------------------------------------
# TAB 1: ACTIVE SCREENING UTILITIES
# ------------------------------------------------------------------------------
with tab_screen:
    st.markdown(f"""
    Upload screening candidates containing structural **SMILES** strings to generate machine learning predictions. 
    All calculations are backed by an automated **Applicability Domain (Apd)** validation metric and **Lipinski Rule of 5** ADME filtration.
    * **Active APD Threshold Limit:** `{APD_THRESHOLD_CONSTANT:.4f}`
    """)

    # --- SINGLE COMPOUND SCREEN ---
    st.write("### 🧪 Single Compound Quick Screen")
    single_smiles = st.text_input("Paste a single SMILES string here (e.g., chloroquine): CCN(CCCC(Nc1c2ccc(Cl)cc2ncc1)C)CC")

    if single_smiles:
        single_smiles = single_smiles.strip()
        fp = smiles_to_ecfp4(single_smiles)
        adme_res = compute_adme_lipinski(single_smiles)
        
        if fp is None or adme_res[4] == "Invalid Structure":
            st.error("❌ Invalid SMILES structure string. Please verify chemical notation.")
        else:
            X_single = np.array([fp])
            
            pred = model.predict(X_single)[0]
            prob = model.predict_proba(X_single)[0][1]
            
            dist, _ = nn_engine.kneighbors(X_single, n_neighbors=5)
            mean_dist = np.mean(dist)
            
            activity_res = "Active" if pred == 1 else "Inactive"
            apd_res = "Reliable" if mean_dist <= APD_THRESHOLD_CONSTANT else "Unreliable"
            
            st.markdown("#### 📊 Screening Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Predicted Class", value=activity_res)
            col2.metric(label="Probability Score", value=f"{prob*100:.1f}%")
            col3.metric(label="APD Status", value=apd_res)
            col4.metric(label="Lipinski ADME Status", value=adme_res[4])
            
            st.markdown("#### 💊 Core Physicochemical Properties")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Mol Weight (MW)", f"{adme_res[0]} Da")
            p2.metric("LogP (Lipophilicity)", adme_res[1])
            p3.metric("H-Bond Donors", adme_res[2])
            p4.metric("H-Bond Acceptors", adme_res[3])

    st.markdown("---")

    # --- BATCH HIGH-THROUGHPUT SCREEN ---
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
                with st.spinner("Streaming chemical spaces, evaluating ADME rules, and processing APD constraints..."):
                    for chunk in pd.read_csv(uploaded_file, chunksize=5000):
                        chunk = chunk.reset_index(drop=True)
                        
                        # High-speed vectorized calculation of ADME parameters
                        adme_data = list(chunk[target_col].apply(compute_adme_lipinski))
                        adme_df = pd.DataFrame(adme_data, columns=['MW (Da)', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 'Lipinski Status'])
                        
                        # Bind properties to core chunk row index
                        chunk = pd.concat([chunk, adme_df], axis=1)
                        
                        fingerprints = chunk[target_col].apply(smiles_to_ecfp4)
                        valid_mask = fingerprints.notna()
                        
                        # Set default placeholder baselines
                        chunk["Activity Prediction"] = "Invalid SMILES structure"
                        chunk["Probability Score"] = "N/A"
                        chunk["Mean Neighbor Distance"] = "N/A"
                        chunk["APD Status"] = "N/A"
                        
                        if valid_mask.any():
                            X_screen = np.array(list(fingerprints[valid_mask]), dtype=np.int8)
                            
                            preds = model.predict(X_screen)
                            probs = model.predict_proba(X_screen)[:, 1]
                            
                            distances, _ = nn_engine.kneighbors(X_screen, n_neighbors=5)
                            mean_distances = np.mean(distances, axis=1)
                            
                            # Vectorized assignment back into chunk dataframe
                            chunk.loc[valid_mask, "Activity Prediction"] = ["Active" if p == 1 else "Inactive" for p in preds]
                            chunk.loc[valid_mask, "Probability Score"] = [f"{prob*100:.1f}%" for prob in probs]
                            chunk.loc[valid_mask, "Mean Neighbor Distance"] = [f"{d:.4f}" for d in mean_distances]
                            chunk.loc[valid_mask, "APD Status"] = ["Reliable" if d <= APD_THRESHOLD_CONSTANT else "Unreliable" for d in mean_distances]
                        
                        processed_chunks.append(chunk)
                        processed_rows += len(chunk)
                        status_text.text(f"Processed structural records: {processed_rows:,}")

                progress_bar.progress(100)
                status_text.text(f"Complete! Total evaluated compounds: {processed_rows:,}")
                
                results_df = pd.concat(processed_chunks, ignore_index=True)
                
                # --- INTERACTIVE DATA FILTRATION PANEL ---
                st.markdown("### 📊 Dataset View Filter Configurations")
                view_selection = st.radio(
                    "Filter displayed data tracking rows:",
                    ["Show All Checked Compounds", "Show Hits Only (Predicted Active)", "Show Orally Bioavailable Hits Only (Active & Lipinski Pass)"],
                    horizontal=True
                )
                
                filtered_df = results_df.copy()
                if view_selection == "Show Hits Only (Predicted Active)":
                    filtered_df = filtered_df[filtered_df["Activity Prediction"] == "Active"]
                elif view_selection == "Show Orally Bioavailable Hits Only (Active & Lipinski Pass)":
                    filtered_df = filtered_df[(filtered_df["Activity Prediction"] == "Active") & (filtered_df["Lipinski Status"] == "Pass")]
                
                st.write(f"Showing **{len(filtered_df):,}** matching compounds:")
                st.dataframe(filtered_df.head(500), use_container_width=True)
                
                csv_export = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sorted Filtering Dataset",
                    data=csv_export,
                    file_name="malaria_adme_filtered_results.csv",
                    mime="text/csv"
                )
                
            except Exception as batch_error:
                st.error("❌ An unexpected pipeline tracking error occurred during processing.")
                st.code(traceback.format_exc(), language="python")

# ------------------------------------------------------------------------------
# TAB 2: MODEL DIAGNOSTICS & DOCUMENTATION
# ------------------------------------------------------------------------------
with tab_metrics:
    st.markdown("### 🧬 Machine Learning Performance Matrices")
    st.write("""
    This web application implements a robust **Random Forest Classifier** optimized to predict inhibitory activity against *Plasmodium falciparum*. 
    Chemical inputs are structurally parsed and featurized into **2048-bit ECFP4 (Extended-Connectivity Fingerprints)** with a bond radius of 2. 
    This model is the updated version of our previously published work:
    """)
    
    st.markdown(":red[Kore, M., Acharya, D., Sharma, L. et al. Development and experimental validation of a machine learning model for the prediction of new antimalarials. BMC Chemistry 19, 28 (2025). https://doi.org/10.1186/s13065-025-01395-4]")
    
    # Dataset Splits Block
    st.markdown("#### 📐 Dataset Stratification")
    col_data1, col_data2, col_data3 = st.columns(3)
    with col_data1:
        st.metric(label="Total Compounds Ensembled", value="15,118")
    with col_data2:
        st.metric(label="Training Set Size (80%)", value="12,094")
    with col_data3:
        st.metric(label="Independent Test Set Size (20%)", value="3,024")

    st.markdown("---")

    # Classification Performance & Metrics Matrix
    st.markdown("#### 🎯 Classification Performance")
    col_perf, col_matrix = st.columns(2)
    
    with col_perf:
        st.write("**Core Statistical Indicators (Test Set Validation):**")
        st.write("- **Area Under the ROC Curve (ROC-AUC):** `0.97` (Highly Discriminative)")
        st.write("- **Sensitivity / Recall (True Active Rate):** `87.3%`")
        st.write("- **Specificity (True Inactive Rate):** `97.0%`")
        st.write("- **Precision (Positive Predictive Value):** `96.2%`")
        st.write("- **Matthews Correlation Coefficient (MCC):** `0.85`")
        
        # Display the ROC Curve Image dynamically
        st.write("**Validation Curve Graphic:**")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="Receiver Operating Characteristic (ROC) Curve - Test Set Evaluation", use_container_width=True)
        else:
            st.warning("⚠️ 'roc_curve.png' file not detected in root directory. Please upload your graphic asset to GitHub.")

    with col_matrix:
        st.write("**Confusion Matrix Contingency Layout:**")
        matrix_data = {
            "Predicted Inactive (0)": ["True Inactives (TN): 1,568", "False Inactives (FN): 179"],
            "Predicted Active (1)": ["False Actives (FP): 48", "True Actives (TP): 1,229"]
        }
        matrix_df = pd.DataFrame(matrix_data, index=["Actual Inactive (0)", "Actual Active (1)"])
        st.dataframe(matrix_df, use_container_width=True)

    st.markdown("---")

    # Informational Guide Rails for App Observers
    st.markdown("#### 🔬 Interpretation")
    st.info("""
    * **ROC-AUC (Receiver Operating Characteristic):** Represents the probability that the model will rank a randomly chosen active compound higher than a randomly chosen inactive one. A value approaching 1.0 defines near-perfect separation properties.
    * **Applicability Domain (APD) Constraint Rules:** Predictive models are vulnerable when analyzing novel chemical families. Our $k$-Nearest Neighbors ($k$-NN) system evaluates geometric structural distances against our chemical space boundary of **0.6744**. Any compound scaling past this distance receives an *Unreliable* designation.
    """)
