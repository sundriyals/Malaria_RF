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
        div[data-testid="stTabs"] button:hover {
            background-color: #e4e7ed;
            border-color: #c0c4cc;
            color: #ff4b4b;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #ff4b4b !important;
            color: white !important;
            border-color: #ff4b4b !important;
            box-shadow: 0px 4px 6px rgba(255, 75, 75, 0.2);
        }
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
# MOLECULAR FEATURIZATION & PROPERTY ENGINE
# ==============================================================================
def smiles_to_ecfp4(smiles, radius=2, nBits=2048):
    """Converts a SMILES string into a 2048-bit binary ECFP4 fingerprint."""
    try:
        smiles = str(smiles).strip()
        if not smiles or smiles == "nan" or smiles.lower() == "none":
            return None
        mol = Chem.MolFromSmiles(smiles)
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
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
        pass_fail = "Pass" if violations <= 1 else "Fail"
        return [round(mw, 2), round(logp, 2), hbd, hba, pass_fail]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan, "Calculation Error"]

def get_datawarrior_aligned_amcs(smiles):
    """Calculates DataWarrior chemical filters via atom-by-atom validation (BaN >= 1, AR >= 2, cLogP >= 2.0, TPSA <= 80.0)"""
    try:
        smiles = str(smiles).strip()
        if not smiles or smiles == "nan" or smiles.lower() == "none":
            return [np.nan, np.nan, np.nan, np.nan, "Invalid Structure"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
        if mol is None:
            return [np.nan, np.nan, np.nan, np.nan, "Invalid Structure"]
            
        clogp = round(Descriptors.MolLogP(mol), 2)
        tpsa = round(Descriptors.TPSA(mol), 2)
        ar = Descriptors.NumAromaticRings(mol)
        
        # Non-basic nitrogen environment exclusions
        amide_smarts = Chem.MolFromSmarts("[NX3][CX3](=[OX1,SX1])")
        sulfonamide_smarts = Chem.MolFromSmarts("[NX3][SX4](=[OX1])(=[OX1])")
        nitro_smarts = Chem.MolFromSmarts("[NX3](=[OX1])=[OX1]")
        
        amide_matches = set(x[0] for x in mol.GetSubstructMatches(amide_smarts)) if amide_smarts else set()
        sulfonamide_matches = set(x[0] for x in mol.GetSubstructMatches(sulfonamide_smarts)) if sulfonamide_smarts else set()
        nitro_matches = set(x[0] for x in mol.GetSubstructMatches(nitro_smarts)) if nitro_smarts else set()
        
        ban = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen check
                idx = atom.GetIdx()
                if idx in amide_matches or idx in sulfonamide_matches or idx in nitro_matches:
                    continue
                
                # Aliphatic Nitrogens (e.g., diethylamine tails)
                if not atom.GetIsAromatic():
                    # Exclude anilines/aryl-amines (nitrogens connected directly to an aromatic ring)
                    if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
                        continue
                    ban += 1
                # Aromatic Nitrogens (e.g., pyridines/quinolines where connection degree == 2)
                else:
                    if atom.GetDegree() == 2:
                        ban += 1
                        
        is_amcs_pass = (ban >= 1) and (ar >= 2) and (clogp >= 2.0) and (tpsa <= 80.0)
        amcs_status = "Pass" if is_amcs_pass else "Fail"
        return [ban, ar, clogp, tpsa, amcs_status]
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
                    if not buffer: break
                    out_file.write(buffer)
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP {e.code} Error downloading artifact: {url}")

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
st.title("🔬 Anti-Plasmodial Activity & Property Profiling Portal")
tab_screen, tab_metrics = st.tabs(["🧪 Screening Portal", "📊 Model Validation & Metrics"])

with tab_screen:
    st.markdown(f"Upload screening candidates containing structural **SMILES** strings to evaluate anti-malarial properties. Compounds are prioritized via ML engines, **Lipinski rules**, and DataWarrior **AMCS parameters**.\n* **Model APD Threshold Boundary:** `{APD_THRESHOLD_CONSTANT:.4f}`")

    st.write("### 🧪 Single Compound Quick Screen")
    single_smiles = st.text_input("Paste a single SMILES string here (e.g., chloroquine):", value="CCN(CCCC(Nc1c2ccc(Cl)cc2ncc1)C)CC")

    if single_smiles:
        single_smiles = single_smiles.strip()
        fp = smiles_to_ecfp4(single_smiles)
        adme_res = compute_adme_lipinski(single_smiles)
        amcs_res = get_datawarrior_aligned_amcs(single_smiles)
        
        if fp is None or adme_res[4] == "Invalid Structure" or amcs_res[4] == "Invalid Structure":
            st.error("❌ Invalid SMILES structure string. Please verify chemical notation.")
        else:
            X_single = np.array([fp])
            pred = model.predict(X_single)[0]
            prob = model.predict_proba(X_single)[0][1]
            dist, _ = nn_engine.kneighbors(X_single, n_neighbors=5)
            mean_dist = np.mean(dist)
            activity_res = "Active" if pred == 1 else "Inactive"
            apd_res = "Reliable" if mean_dist <= APD_THRESHOLD_CONSTANT else "Unreliable"
            
            st.markdown("#### 📊 Core Status Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Predicted Class", activity_res)
            col2.metric("Probability Score", f"{prob*100:.1f}%")
            col3.metric("Model APD Domain", apd_res)
            col4.metric("Lipinski Status", adme_res[4])
            col5.metric("AMCS Space Status", amcs_res[4])
            
            st.markdown("#### 💊 Physicochemical Descriptor Profile Breakdown")
            p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
            p1.metric("Mol Weight", f"{adme_res[0]} Da")
            p2.metric("Lipophilicity (LogP)", adme_res[1])
            p3.metric("H-Bond Donors", adme_res[2])
            p4.metric("H-Bond Acceptors", adme_res[3])
            p5.metric("Aromatic Rings", int(amcs_res[1]) if not np.isnan(amcs_res[1]) else "N/A")
            p6.metric("Basic Nitrogens (BaN)", int(amcs_res[0]) if not np.isnan(amcs_res[0]) else "N/A")
            p7.metric("TPSA Surface Area", f"{amcs_res[3]} Å²" if not np.isnan(amcs_res[3]) else "N/A")

    st.markdown("---")
    st.write("### 📂 Batch File High-Throughput Screening")
    uploaded_file = st.file_uploader("Choose a CSV file to screen", type=["csv"])

    if uploaded_file is not None:
        try:
            header_df = pd.read_csv(uploaded_file, nrows=2)
            uploaded_file.seek(0) 
        except Exception as e:
            st.error(f"❌ Read Error: {e}"); st.stop()

        smiles_col = [col for col in header_df.columns if col.lower() in ['smiles', 'smiles string', 'structure']]
        if not smiles_col:
            st.error("❌ Column Error: Could not locate a 'Smiles' or 'Structure' column header.")
        else:
            target_col = smiles_col[0]
            st.success(f"Processing structural pipeline using column: '{target_col}'")
            processed_chunks, progress_bar = [], st.progress(0)
            status_text, processed_rows = st.empty(), 0
            
            try:
                with st.spinner("Streaming data matrices, checking ADME models..."):
                    for chunk in pd.read_csv(uploaded_file, chunksize=5000):
                        chunk = chunk.reset_index(drop=True)
                        adme_data = list(chunk[target_col].apply(compute_adme_lipinski))
                        adme_df = pd.DataFrame(adme_data, columns=['MW (Da)', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 'Lipinski Status'])
                        amcs_data = list(chunk[target_col].apply(get_datawarrior_aligned_amcs))
                        amcs_df = pd.DataFrame(amcs_data, columns=['Basic Nitrogens', 'Aromatic Rings', 'cLogP', 'TPSA (Å²)', 'AMCS Status'])
                        chunk = pd.concat([chunk, adme_df, amcs_df], axis=1)
                        fingerprints = chunk[target_col].apply(smiles_to_ecfp4)
                        valid_mask = fingerprints.notna()
                        chunk["Activity Prediction"], chunk["Probability Score"] = "Invalid Structure", "N/A"
                        chunk["Mean Neighbor Distance"], chunk["Model APD Status"] = "N/A", "N/A"
                        
                        if valid_mask.any():
                            X_screen = np.array(list(fingerprints[valid_mask]), dtype=np.int8)
                            preds = model.predict(X_screen)
                            probs = model.predict_proba(X_screen)[:, 1]
                            distances, _ = nn_engine.kneighbors(X_screen, n_neighbors=5)
                            mean_distances = np.mean(distances, axis=1)
                            chunk.loc[valid_mask, "Activity Prediction"] = ["Active" if p == 1 else "Inactive" for p in preds]
                            chunk.loc[valid_mask, "Probability Score"] = [f"{prob*100:.1f}%" for prob in probs]
                            chunk.loc[valid_mask, "Mean Neighbor Distance"] = [f"{d:.4f}" for d in mean_distances]
                            chunk.loc[valid_mask, "Model APD Status"] = ["Reliable" if d <= APD_THRESHOLD_CONSTANT else "Unreliable" for d in mean_distances]
                        
                        processed_chunks.append(chunk)
                        processed_rows += len(chunk)
                        status_text.text(f"Processed records: {processed_rows:,}")

                progress_bar.progress(100)
                results_df = pd.concat(processed_chunks, ignore_index=True)
                
                st.markdown("### 📊 Dataset View Filter Configurations")
                view_selection = st.radio("Filter output rows:", ["Show All Checked Compounds", "Show Hits Only (Predicted Active)", "Show Active & Lipinski Pass Only", "Show Active & AMCS Space Pass Only", "Show Elite Leads Only (Active, Lipinski Pass, & AMCS Pass)"], horizontal=True)
                
                filtered_df = results_df.copy()
                if view_selection == "Show Hits Only (Predicted Active)":
                    filtered_df = filtered_df[filtered_df["Activity Prediction"] == "Active"]
                elif view_selection == "Show Active & Lipinski Pass Only":
                    filtered_df = filtered_df[(filtered_df["Activity Prediction"] == "Active") & (filtered_df["Lipinski Status"] == "Pass")]
                elif view_selection == "Show Active & AMCS Space Pass Only":
                    filtered_df = filtered_df[(filtered_df["Activity Prediction"] == "Active") & (filtered_df["AMCS Status"] == "Pass")]
                elif view_selection == "Show Elite Leads Only (Active, Lipinski Pass, & AMCS Pass)":
                    filtered_df = filtered_df[(filtered_df["Activity Prediction"] == "Active") & (filtered_df["Lipinski Status"] == "Pass") & (filtered_df["AMCS Status"] == "Pass")]
                
                st.write(f"Showing **{len(filtered_df):,}** matching compounds:")
                st.dataframe(filtered_df.head(500), use_container_width=True)
                st.download_button("📥 Download Export Dataset", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name="malaria_leads_output.csv", mime="text/csv")
            except Exception as batch_error:
                st.error("❌ Processing pipeline error."); st.code(traceback.format_exc(), language="python")

with tab_metrics:
    st.markdown("### 🧬 Machine Learning Performance Matrices")
    st.write("This web application implements a Random Forest Classifier featurized into **2048-bit ECFP4 fingerprints**. This is the updated implementation of our work:")
    st.markdown(":red[Kore, M., Acharya, D., Sharma, L. et al. Development and experimental validation of a machine learning model for the prediction of new antimalarials. BMC Chemistry 19, 28 (2025).]")
    
    col_data1, col_data2, col_data3 = st.columns(3)
    col_data1.metric("Total Compounds Ensembled", "15,118")
    col_data2.metric("Training Set Size (80%)", "12,094")
    col_data3.metric("Independent Test Set (20%)", "3,024")

    st.markdown("---")
    col_perf, col_matrix = st.columns(2)
    with col_perf:
        st.write("**Core Metrics Evaluation:**\n- **ROC-AUC Score:** `0.97` (Highly Discriminative)\n- **Sensitivity (Recall):** `87.3%`\n- **Specificity:** `97.0%`\n- **Precision:** `96.2%`\n- **Matthews Correlation Coefficient (MCC):** `0.85`")
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="ROC Curve", use_container_width=True)
        else:
            st.warning("⚠️ 'roc_curve.png' graphic file missing from working repository paths.")

    with col_matrix:
        st.write("**Confusion Matrix Layout:**")
        matrix_df = pd.DataFrame({"Predicted Inactive (0)": ["True Inactives (TN): 1,568", "False Inactives (FN): 179"], "Predicted Active (1)": ["False Actives (FP): 48", "True Actives (TP): 1,229"]}, index=["Actual Inactive (0)", "Actual Active (1)"])
        st.dataframe(matrix_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 Interpretation Definitions")
    st.info(r"""
    * **Lipinski's Rule of 5 (ADME Validation):** Evaluates overall structural drug-likeness based on basic pharmacokinetic property parameters (MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10). Allows 1 threshold violation.
    * **Antimalarial Chemical Space (AMCS Filters):** Implements specialized rule sets matching DataWarrior's logic optimization for identifying specific antimalarial properties (Basic Nitrogens >= 1, Aromatic Rings >= 2, cLogP >= 2.0, TPSA <= 80 Å²). 
    """)
