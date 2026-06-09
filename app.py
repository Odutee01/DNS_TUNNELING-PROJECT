import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="DNS Tunneling Detection", layout="wide", page_icon="🛡️")

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* Global Fonts & Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Clean container styling */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.05);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Premium Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        box-shadow: 0 4px 15px rgba(42, 82, 152, 0.25);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.35);
        transform: translateY(-1px);
    }
    
    /* Download button styling */
    div.stDownloadButton > button:first-child {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.25);
        transition: all 0.3s ease;
    }
    div.stDownloadButton > button:first-child:hover {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
        box-shadow: 0 6px 20px rgba(56, 239, 125, 0.35);
        transform: translateY(-1px);
    }
    
    /* Card headers */
    .section-card {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate cards
def get_metric_card(title, value, color, icon):
    return f"""
    <div style="
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-left: 5px solid {color};
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 15px;
    ">
        <div style="font-size: 2.2rem; min-width: 40px; text-align: center;">{icon}</div>
        <div>
            <div style="font-size: 0.8rem; opacity: 0.7; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</div>
            <div style="font-size: 1.6rem; font-weight: 800; margin-top: 2px; line-height: 1;">{value}</div>
        </div>
    </div>
    """

# ----------------- MODEL LOADING & SCALER SETUP -----------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the XGBoost model. Make sure `xgboost_model.pkl` is in the directory. Details: {e}")
    st.stop()

@st.cache_resource
def get_scaler_and_features():
    dataset_path = "preprocessed10_dataset.csv"
    if os.path.exists(dataset_path):
        base_df = pd.read_csv(dataset_path)
        X_base = base_df.drop('label', axis=1) if 'label' in base_df.columns else base_df
        X_base_num = X_base.select_dtypes(include=['int64', 'float64'])
        scaler = StandardScaler()
        scaler.fit(X_base_num)
        return scaler, X_base_num.columns.tolist()
    return None, None

scaler, feature_cols = get_scaler_and_features()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <span style="font-size: 3.5rem;">🛡️</span>
        <h2 style="margin: 10px 0 5px 0; font-family: 'Outfit'; font-weight: 800; font-size: 1.5rem;">Model Registry</h2>
        <p style="font-size: 0.85rem; opacity: 0.7;">System & Classifier Configuration</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model Metadata Card
    st.markdown("""
    <div style="
        background-color: rgba(30, 60, 114, 0.1); 
        padding: 1rem; 
        border-radius: 8px; 
        border: 1px solid rgba(30, 60, 114, 0.2);
        margin-bottom: 1.5rem;
    ">
        <h4 style="margin:0 0 5px 0; color: #2a5298; font-size: 1rem;">Active Model: XGBoost</h4>
        <p style="margin:0; font-size:0.8rem; opacity:0.8;"><b>Type:</b> Gradient Boosted Trees</p>
        <p style="margin:0; font-size:0.8rem; opacity:0.8;"><b>Accuracy:</b> ~99.9% (Trained)</p>
    </div>
    """, unsafe_allow_html=True)

    if scaler is None:
        st.error("⚠️ preprocessed10_dataset.csv not found! Standard scaling will be disabled.")
    else:
        st.success("✅ Scaler active & fitted.")

    st.markdown("""
    <div style="
        font-size: 0.8rem; 
        opacity: 0.7; 
        line-height: 1.4;
        background-color: rgba(128,128,128,0.05); 
        padding: 10px; 
        border-radius: 8px;
        margin-top: 2rem;
    ">
        <b>About DNS Tunneling:</b><br/>
        Malicious users wrap non-DNS protocols inside DNS queries to bypass firewalls and exfiltrate data. This tool uses machine learning to detect these patterns based on traffic features.
    </div>
    """, unsafe_allow_html=True)

# ----------------- MAIN LAYOUT -----------------

# Header Banner
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
    padding: 2.5rem; 
    border-radius: 16px; 
    margin-bottom: 2rem; 
    color: white; 
    box-shadow: 0 10px 30px rgba(42,82,152,0.15)
">
    <h1 style="margin: 0; font-family: 'Outfit', sans-serif; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;">DNS Tunneling Detection</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem; font-weight: 300;">Real-time packet structure analysis using a high-performance XGBoost model.</p>
</div>
""", unsafe_allow_html=True)

# Instructions & Upload Container
col_setup, col_preview = st.columns([2, 3], gap="large")

with col_setup:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Upload Dataset")
    st.write("Upload a CSV file containing DNS traffic logs. The app will align features, perform normalization, and classify each query.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# Process Uploaded File
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    original_df_display = df.copy()
    
    # Pre-clean label column if exists in user-uploaded file (to avoid leakage)
    if 'label' in df.columns:
        df = df.drop('label', axis=1)

    with col_preview:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📋 Dataset Preview")
        st.write(f"Loaded **{df.shape[0]}** rows and **{df.shape[1]}** columns.")
        st.dataframe(df.head(5), use_container_width=True)
        
        analyze_btn = st.button("🚀 Analyze Traffic & Detect Anomalies")
        st.markdown('</div>', unsafe_allow_html=True)

    if 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Classifying DNS traffic patterns via XGBoost..."):
            try:
                # Prepare data
                X_new = df.copy()
                    
                # Align with training features
                if feature_cols:
                    for col in feature_cols:
                        if col not in X_new.columns:
                            X_new[col] = 0
                    X_new = X_new[feature_cols] # Ensure order
                    X_scaled = scaler.transform(X_new)
                else:
                    X_new = X_new.select_dtypes(include=['int64', 'float64'])
                    X_scaled = X_new.values

                # Predict
                predictions = model.predict(X_scaled)
                
                # Assemble Results
                results_df = original_df_display.copy()
                results_df['Prediction_Class'] = predictions
                results_df['Prediction_Label'] = results_df['Prediction_Class'].map({0: 'Benign (Normal)', 1: 'Attack (Tunneling)'})
                
                # Compute Stats
                total_cnt = len(predictions)
                attack_cnt = int(sum(predictions))
                benign_cnt = total_cnt - attack_cnt
                attack_pct = (attack_cnt / total_cnt) * 100
                
                st.markdown("<br/>", unsafe_allow_html=True)
                st.subheader("📊 Detection Dashboard")
                
                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.markdown(get_metric_card("Total Analyzed", f"{total_cnt:,}", "#2a5298", "🔍"), unsafe_allow_html=True)
                with m_col2:
                    st.markdown(get_metric_card("Benign Traffic", f"{benign_cnt:,}", "#2e7d32", "✅"), unsafe_allow_html=True)
                with m_col3:
                    st.markdown(get_metric_card("Attacks Detected", f"{attack_cnt:,}", "#c62828", "🚨"), unsafe_allow_html=True)
                with m_col4:
                    st.markdown(get_metric_card("Threat Ratio", f"{attack_pct:.1f}%", "#ef6c00", "📈"), unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Visualization and Detail Breakdown
                vis_col, data_col = st.columns([2, 3], gap="large")
                
                with vis_col:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.subheader("Distribution Breakdown")
                    chart_df = pd.DataFrame({
                        'Count': [benign_cnt, attack_cnt]
                    }, index=['Benign (Normal)', 'Attack (Tunneling)'])
                    st.bar_chart(chart_df, color="#2a5298" if attack_cnt < benign_cnt else "#c62828")
                    
                    # Custom progress bar for threat ratio
                    st.write("Threat Level Indicator:")
                    st.progress(attack_pct / 100.0)
                    if attack_pct > 20:
                        st.markdown("<span style='color:#c62828; font-weight:bold;'>🚨 High Threat Level Detected!</span> Action is recommended.", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#2e7d32; font-weight:bold;'>✅ Normal Threat Levels.</span> DNS traffic is healthy.", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with data_col:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.subheader("Actionable Insights")
                    st.write("Here is the prediction results dataframe. Scroll to the rightmost columns (`Prediction_Class` & `Prediction_Label`) to see specific classifications.")
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Detailed Reports (CSV)",
                        data=csv,
                        file_name='dns_tunneling_predictions.csv',
                        mime='text/csv',
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
