import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🌱 Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# LANGUAGE TRANSLATIONS
# ============================================================
TRANSLATIONS = {
    "en": {
        # Header
        "app_title": "🌱 Fertilizer Recommendation System",
        "app_subtitle": "AI-powered fertilizer recommendations based on soil conditions and crop type",
        
        # Sidebar sections
        "language": "🌐 Language",
        "env_conditions": "🌍 Environmental Conditions",
        "temperature": "🌡️ Temperature (°C)",
        "temperature_help": "Current temperature in Celsius",
        "humidity": "💧 Humidity (%)",
        "humidity_help": "Relative humidity percentage",
        "moisture": "🌊 Soil Moisture (%)",
        "moisture_help": "Soil moisture content",
        "soil_crop_info": "🌾 Soil & Crop Information",
        "soil_type": "🏔️ Soil Type",
        "crop_type": "🌿 Crop Type",
        "soil_nutrients": "🧪 Soil Nutrients (mg/kg)",
        "nitrogen": "🟢 Nitrogen (N)",
        "nitrogen_help": "Nitrogen content in soil",
        "potassium": "🟠 Potassium (K)",
        "potassium_help": "Potassium content in soil",
        "phosphorous": "🔵 Phosphorous (P)",
        "phosphorous_help": "Phosphorous content in soil",
        "get_recommendation": "🔍 Get Recommendation",
        
        # Main content
        "input_summary": "📊 Current Input Summary",
        "nutrient_levels": "🧪 Soil Nutrient Levels",
        "current_selection": "🌱 Current Selection",
        "crop": "Crop",
        "soil": "Soil",
        "npk_status": "🧪 NPK Status",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        
        # Results
        "analyzing": "🔄 Analyzing soil conditions and generating recommendation...",
        "recommendation_result": "🎯 Analysis Result",
        "recommended_fertilizer": "Recommended Fertilizer",
        "confidence": "Confidence",
        "npk_ratio": "NPK Ratio",
        "fertilizer_details": "📋 Fertilizer Details",
        "description": "📝 Description",
        "best_for": "🎯 Best For",
        "application_tips": "💡 Application Tips",
        "confidence_dist": "📊 Prediction Confidence Distribution",
        "error_prediction": "❌ Error making prediction",
        
        # Footer
        "quick_tips": "💡 Quick Tips for Fertilizer Application",
        "tip1": "📌 Always conduct soil tests before applying fertilizers",
        "tip2": "📌 Follow recommended dosage to avoid over-fertilization",
        "tip3": "📌 Apply fertilizers during cooler parts of the day",
        "tip4": "📌 Ensure adequate soil moisture for nutrient absorption",
        "made_with": "Made with ❤️ using Streamlit & Machine Learning",
        
        # Error
        "models_not_loaded": "⚠️ Models not loaded. Please run `train_fertilizer_model_v2.py` first to train the model.",
        
        # Chart labels
        "content_mg": "Content (mg/kg)",
        "confidence_pct": "Confidence (%)",
        "fertilizer": "Fertilizer",
        "probability": "Probability"
    },
    "ta": {
        # Header
        "app_title": "🌱 உர பரிந்துரை அமைப்பு",
        "app_subtitle": "மண் நிலை மற்றும் பயிர் வகையின் அடிப்படையில் AI-இயங்கும் உர பரிந்துரைகள்",
        
        # Sidebar sections
        "language": "🌐 மொழி",
        "env_conditions": "🌍 சுற்றுச்சூழல் நிலைகள்",
        "temperature": "🌡️ வெப்பநிலை (°C)",
        "temperature_help": "தற்போதைய வெப்பநிலை செல்சியஸில்",
        "humidity": "💧 ஈரப்பதம் (%)",
        "humidity_help": "சார்பு ஈரப்பதம் சதவீதம்",
        "moisture": "🌊 மண் ஈரப்பதம் (%)",
        "moisture_help": "மண் ஈரப்பத உள்ளடக்கம்",
        "soil_crop_info": "🌾 மண் & பயிர் தகவல்",
        "soil_type": "🏔️ மண் வகை",
        "crop_type": "🌿 பயிர் வகை",
        "soil_nutrients": "🧪 மண் ஊட்டச்சத்துகள் (mg/kg)",
        "nitrogen": "🟢 நைட்ரஜன் (N)",
        "nitrogen_help": "மண்ணில் நைட்ரஜன் உள்ளடக்கம்",
        "potassium": "🟠 பொட்டாசியம் (K)",
        "potassium_help": "மண்ணில் பொட்டாசியம் உள்ளடக்கம்",
        "phosphorous": "🔵 பாஸ்பரஸ் (P)",
        "phosphorous_help": "மண்ணில் பாஸ்பரஸ் உள்ளடக்கம்",
        "get_recommendation": "🔍 பரிந்துரை பெறுக",
        
        # Main content
        "input_summary": "📊 தற்போதைய உள்ளீடு சுருக்கம்",
        "nutrient_levels": "🧪 மண் ஊட்டச்சத்து அளவுகள்",
        "current_selection": "🌱 தற்போதைய தேர்வு",
        "crop": "பயிர்",
        "soil": "மண்",
        "npk_status": "🧪 NPK நிலை",
        "high": "அதிகம்",
        "medium": "நடுத்தரம்",
        "low": "குறைவு",
        
        # Results
        "analyzing": "🔄 மண் நிலைகளை பகுப்பாய்வு செய்து பரிந்துரை உருவாக்குகிறது...",
        "recommendation_result": "🎯 பரிந்துரை முடிவு",
        "recommended_fertilizer": "பரிந்துரைக்கப்பட்ட உரம்",
        "confidence": "நம்பகத்தன்மை",
        "npk_ratio": "NPK விகிதம்",
        "fertilizer_details": "📋 உர விவரங்கள்",
        "description": "📝 விளக்கம்",
        "best_for": "🎯 சிறந்தது",
        "application_tips": "💡 பயன்பாட்டு குறிப்புகள்",
        "confidence_dist": "📊 கணிப்பு நம்பகத்தன்மை விநியோகம்",
        "error_prediction": "❌ கணிப்பில் பிழை",
        
        # Footer
        "quick_tips": "💡 உர பயன்பாட்டிற்கான விரைவு குறிப்புகள்",
        "tip1": "📌 உரங்களை பயன்படுத்துவதற்கு முன் எப்போதும் மண் பரிசோதனைகளை நடத்தவும்",
        "tip2": "📌 அதிக உரமிடுவதைத் தவிர்க்க பரிந்துரைக்கப்பட்ட அளவைப் பின்பற்றவும்",
        "tip3": "📌 நாளின் குளிர்ச்சியான நேரத்தில் உரங்களை இடவும்",
        "tip4": "📌 ஊட்டச்சத்து உறிஞ்சுதலுக்கு போதுமான மண் ஈரப்பதத்தை உறுதி செய்யவும்",
        "made_with": "Streamlit & Machine Learning மூலம் ❤️ உடன் உருவாக்கப்பட்டது",
        
        # Error
        "models_not_loaded": "⚠️ மாதிரிகள் ஏற்றப்படவில்லை. முதலில் `train_fertilizer_model_v2.py` இயக்கவும்.",
        
        # Chart labels
        "content_mg": "உள்ளடக்கம் (mg/kg)",
        "confidence_pct": "நம்பகத்தன்மை (%)",
        "fertilizer": "உரம்",
        "probability": "நிகழ்தகவு"
    }
}

# Fertilizer info with translations
FERTILIZER_INFO = {
    "en": {
        "No Fertilizer Needed": {
            "description": "Your soil is perfectly rich in primary nutrients! No additional fertilizers are needed.",
            "npk": "0-0-0",
            "best_for": "Maintaining current optimal soil health",
            "application": "Continue regular watering and monitoring"
        },
        "Urea": {
            "description": "High nitrogen fertilizer (46% N) - ideal for leafy growth and green color",
            "npk": "46-0-0",
            "best_for": "Nitrogen-deficient soils, leafy vegetables, cereals",
            "application": "Apply in split doses during active growth phase"
        },
        "DAP": {
            "description": "Di-Ammonium Phosphate - provides nitrogen and phosphorus for root development",
            "npk": "18-46-0",
            "best_for": "Root development, flowering, fruit setting",
            "application": "Apply at planting time, mix with soil"
        },
        "14-35-14": {
            "description": "Balanced NPK fertilizer with high phosphorus content",
            "npk": "14-35-14",
            "best_for": "Flowering plants, fruiting vegetables",
            "application": "Apply during pre-flowering stage"
        },
        "28-28": {
            "description": "High nitrogen-phosphorus fertilizer for balanced growth",
            "npk": "28-28-0",
            "best_for": "General crop nutrition, cereals",
            "application": "Apply as basal dose"
        },
        "17-17-17": {
            "description": "Perfectly balanced NPK fertilizer for overall plant health",
            "npk": "17-17-17",
            "best_for": "All-purpose use, mixed cropping",
            "application": "Apply as needed throughout growing season"
        },
        "20-20": {
            "description": "Balanced nitrogen-phosphorus fertilizer",
            "npk": "20-20-0",
            "best_for": "Cereals, pulses, oilseeds",
            "application": "Apply at sowing and flowering stages"
        },
        "10-26-26": {
            "description": "High phosphorus and potassium fertilizer for fruiting and disease resistance",
            "npk": "10-26-26",
            "best_for": "Fruit quality, disease resistance, root crops",
            "application": "Apply during fruiting and maturity stages"
        }
    },
    "ta": {
        "No Fertilizer Needed": {
            "description": "உங்கள் மண்ணில் ஏற்கனவே அத்தியாவசிய ஊட்டச்சத்துக்கள் நிறைந்துள்ளன! உரம் தேவையில்லை.",
            "npk": "0-0-0",
            "best_for": "தற்போதைய சிறந்த மண் ஆரோக்கியத்தை பராமரிக்க",
            "application": "வழக்கமான நீர்ப்பாசனம் மற்றும் கண்காணிப்பை தொடரவும்"
        },
        "Urea": {
            "description": "அதிக நைட்ரஜன் உரம் (46% N) - இலை வளர்ச்சி மற்றும் பச்சை நிறத்திற்கு ஏற்றது",
            "npk": "46-0-0",
            "best_for": "நைட்ரஜன் குறைபாடுள்ள மண், இலைக் காய்கறிகள், தானியங்கள்",
            "application": "செயலில் வளர்ச்சி கட்டத்தில் பிரித்து இடவும்"
        },
        "DAP": {
            "description": "டை-அமோனியம் பாஸ்பேட் - வேர் வளர்ச்சிக்கு நைட்ரஜன் மற்றும் பாஸ்பரஸ் வழங்குகிறது",
            "npk": "18-46-0",
            "best_for": "வேர் வளர்ச்சி, பூக்கும், பழம் பிடிக்கும்",
            "application": "நடும் நேரத்தில் மண்ணுடன் கலந்து இடவும்"
        },
        "14-35-14": {
            "description": "அதிக பாஸ்பரஸ் உள்ளடக்கத்துடன் சமநிலை NPK உரம்",
            "npk": "14-35-14",
            "best_for": "பூக்கும் செடிகள், பழ காய்கறிகள்",
            "application": "பூக்கும் முன் நிலையில் இடவும்"
        },
        "28-28": {
            "description": "சமநிலை வளர்ச்சிக்கான அதிக நைட்ரஜன்-பாஸ்பரஸ் உரம்",
            "npk": "28-28-0",
            "best_for": "பொது பயிர் ஊட்டச்சத்து, தானியங்கள்",
            "application": "அடிஉரமாக இடவும்"
        },
        "17-17-17": {
            "description": "ஒட்டுமொத்த தாவர ஆரோக்கியத்திற்கான சரியான சமநிலை NPK உரம்",
            "npk": "17-17-17",
            "best_for": "அனைத்து நோக்கத்திற்கும், கலப்பு பயிர்",
            "application": "வளரும் பருவம் முழுவதும் தேவைக்கேற்ப இடவும்"
        },
        "20-20": {
            "description": "சமநிலை நைட்ரஜன்-பாஸ்பரஸ் உரம்",
            "npk": "20-20-0",
            "best_for": "தானியங்கள், பயறு வகைகள், எண்ணெய் வித்துக்கள்",
            "application": "விதைப்பு மற்றும் பூக்கும் நிலைகளில் இடவும்"
        },
        "10-26-26": {
            "description": "பழ தரம் மற்றும் நோய் எதிர்ப்புக்கான அதிக பாஸ்பரஸ் மற்றும் பொட்டாசியம் உரம்",
            "npk": "10-26-26",
            "best_for": "பழ தரம், நோய் எதிர்ப்பு, கிழங்கு பயிர்கள்",
            "application": "பழம் பிடிக்கும் மற்றும் முதிர்ச்சி நிலைகளில் இடவும்"
        }
    }
}

# Custom CSS for dark theme (works in both light and dark mode)
st.markdown("""
<style>
    /* Force dark theme on entire app */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    }
    
    /* Main content area */
    .main .block-container {
        background: transparent !important;
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #16213e 0%, #1a1a2e 100%) !important;
    }
    
    /* All text white/light */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #e8e8e8 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        background: rgba(30, 58, 95, 0.5) !important;
        border-radius: 10px;
        padding: 10px !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #00b09b !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a8dadc !important;
    }
    
    /* Selectbox and sliders */
    .stSelectbox > div > div, .stSlider > div {
        background: rgba(30, 58, 95, 0.3) !important;
        border-radius: 8px;
    }
    
    .stSelectbox label, .stSlider label {
        color: #a8dadc !important;
        font-weight: 600 !important;
    }
    
    /* Slider track and thumb */
    .stSlider [data-baseweb="slider"] {
        background: transparent !important;
    }
    
    /* Info, Success, Warning boxes */
    .stAlert, [data-testid="stAlert"] {
        background: rgba(30, 58, 95, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    .stAlert p, [data-testid="stAlert"] p {
        color: #e8e8e8 !important;
    }
    
    /* Success box specific */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background: rgba(0, 176, 155, 0.2) !important;
        border-left: 4px solid #00b09b !important;
    }
    
    /* Info box specific */
    .stInfo, [data-baseweb="notification"][kind="info"] {
        background: rgba(33, 150, 243, 0.2) !important;
        border-left: 4px solid #2196F3 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #e8e8e8 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #a8dadc !important;
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        color: #a8dadc !important;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1e3a5f, #16213e) !important;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem auto;
        max-width: 600px;
    }
    
    .fertilizer-name {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-badge {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white !important;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00b09b, #96c93d) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 176, 155, 0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00b09b !important;
    }
    
    /* Horizontal rule */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot, .plotly {
        background: transparent !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 58, 95, 0.5) !important;
        color: #a8dadc !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00b09b;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #96c93d;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; padding: 0.5rem 0; }
        .sub-header { font-size: 0.9rem; margin-bottom: 1rem; }
        .result-card { padding: 1.2rem; }
        .fertilizer-name { font-size: 1.8rem; }
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        .stButton > button { padding: 0.5rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)


# Load model and encoders
@st.cache_resource
def load_models():
    try:
        model = joblib.load('fertilizer_model_v2.joblib')
        scaler = joblib.load('fertilizer_scaler_v2.joblib')
        le_soil = joblib.load('soil_encoder_v2.joblib')
        le_crop = joblib.load('crop_encoder_v2.joblib')
        le_fertilizer = joblib.load('fertilizer_encoder_v2.joblib')
        return model, scaler, le_soil, le_crop, le_fertilizer, True
    except Exception as e:
        return None, None, None, None, None, False

model, scaler, le_soil, le_crop, le_fertilizer, models_loaded = load_models()

# ============================================================
# LANGUAGE SELECTION AND HEADER
# ============================================================
lang_col1, lang_col2 = st.columns([5, 1])
with lang_col2:
    lang = st.selectbox(
        "🌐 Language / மொழி",
        options=["English", "தமிழ் (Tamil)"],
        index=0
    )
    lang_code = "en" if lang == "English" else "ta"
    t = TRANSLATIONS[lang_code]

# Header
st.markdown(f'<h1 class="main-header">{t["app_title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t["app_subtitle"]}</p>', unsafe_allow_html=True)

if not models_loaded:
    st.error(t["models_not_loaded"])
    st.stop()

# Main Inputs section instead of Sidebar
input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    st.markdown(f"### {t['env_conditions']}")
    
    temperature = st.slider(
        t["temperature"],
        min_value=20, max_value=45, value=28,
        help=t["temperature_help"]
    )
    
    humidity = st.slider(
        t["humidity"],
        min_value=40, max_value=90, value=60,
        help=t["humidity_help"]
    )
    
    moisture = st.slider(
        t["moisture"],
        min_value=20, max_value=80, value=40,
        help=t["moisture_help"]
    )

with input_col2:
    st.markdown(f"### {t['soil_nutrients']}")
    
    nitrogen = st.slider(
        t["nitrogen"],
        min_value=0, max_value=50, value=20,
        help=t["nitrogen_help"]
    )
    
    potassium = st.slider(
        t["potassium"],
        min_value=0, max_value=25, value=5,
        help=t["potassium_help"]
    )
    
    phosphorous = st.slider(
        t["phosphorous"],
        min_value=0, max_value=50, value=15,
        help=t["phosphorous_help"]
    )

with input_col3:
    st.markdown(f"### {t['soil_crop_info']}")
    
    soil_types = list(le_soil.classes_)
    soil_type = st.selectbox(
        t["soil_type"],
        options=soil_types,
        index=soil_types.index("Loamy") if "Loamy" in soil_types else 0
    )
    
    crop_types = list(le_crop.classes_)
    crop_type = st.selectbox(
        t["crop_type"],
        options=crop_types,
        index=crop_types.index("Wheat") if "Wheat" in crop_types else 0
    )

st.markdown("---")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### {t['input_summary']}")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(t["temperature"].split("(")[0].strip(), f"{temperature}°C")
    with metric_cols[1]:
        st.metric(t["humidity"].split("(")[0].strip(), f"{humidity}%")
    with metric_cols[2]:
        st.metric(t["moisture"].split("(")[0].strip(), f"{moisture}%")
    with metric_cols[3]:
        st.metric(t["soil"], soil_type)
    
    # NPK visualization
    st.markdown(f"### {t['nutrient_levels']}")
    
    npk_labels = [
        t["nitrogen"].split("(")[0].strip(), 
        t["phosphorous"].split("(")[0].strip(), 
        t["potassium"].split("(")[0].strip()
    ]
    
    npk_fig = go.Figure()
    npk_fig.add_trace(go.Bar(
        x=npk_labels,
        y=[nitrogen, phosphorous, potassium],
        marker_color=['#4CAF50', '#2196F3', '#FF9800'],
        text=[nitrogen, phosphorous, potassium],
        textposition='outside'
    ))
    
    npk_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#a8dadc',
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title=t["content_mg"], gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(npk_fig, width="stretch")

with col2:
    st.markdown(f"### {t['current_selection']}")
    
    st.success(f"**{t['crop']}:** {crop_type}")
    st.info(f"**{t['soil']}:** {soil_type}")
    
    st.markdown(f"#### {t['npk_status']}")
    
    n_status = f"🟢 {t['high']}" if nitrogen > 30 else (f"🟡 {t['medium']}" if nitrogen > 15 else f"🔴 {t['low']}")
    p_status = f"🟢 {t['high']}" if phosphorous > 30 else (f"🟡 {t['medium']}" if phosphorous > 15 else f"🔴 {t['low']}")
    k_status = f"🟢 {t['high']}" if potassium > 15 else (f"🟡 {t['medium']}" if potassium > 5 else f"🔴 {t['low']}")
    
    n_label = t["nitrogen"].replace("🟢 ", "").split("(")[0].strip()
    p_label = t["phosphorous"].replace("🔵 ", "").split("(")[0].strip()
    k_label = t["potassium"].replace("🟠 ", "").split("(")[0].strip()
    
    st.markdown(f"""
    - **{n_label}:** {n_status} ({nitrogen} mg/kg)
    - **{p_label}:** {p_status} ({phosphorous} mg/kg)
    - **{k_label}:** {k_status} ({potassium} mg/kg)
    """)

# Prediction section
st.markdown("---")

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_button = st.button(t["get_recommendation"], width="stretch")
st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    with st.spinner(t["analyzing"]):
        try:
            if nitrogen >= 40 and phosphorous >= 40 and potassium >= 20:
                fertilizer_name = "No Fertilizer Needed"
                confidence = 100.0
                show_charts = False
            else:
                input_data = np.array([[
                    temperature, humidity, moisture,
                    le_soil.transform([soil_type])[0],
                    le_crop.transform([crop_type])[0],
                    nitrogen, potassium, phosphorous
                ]])
                
                # Use DataFrame with exact feature names to avoid StandardScaler warning
                feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                               'Nitrogen', 'Potassium', 'Phosphorous']
                input_df = pd.DataFrame(input_data, columns=feature_cols)
                
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                fertilizer_name = le_fertilizer.inverse_transform(prediction)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_scaled)[0]
                    confidence = np.max(probabilities) * 100
                else:
                    confidence = 95.0
                show_charts = True
            
            st.markdown(f"## {t['recommendation_result']}")
            
            fert_info = FERTILIZER_INFO[lang_code].get(fertilizer_name, {})
            
            st.markdown(f"""
            <div class="result-card">
                <p style="color: #a8dadc; font-size: 1.1rem; text-align: center;">{t['recommended_fertilizer']}</p>
                <h2 class="fertilizer-name">🌿 {fertilizer_name}</h2>
                <p style="text-align: center;">
                    <span class="confidence-badge">{t['confidence']}: {confidence:.1f}%</span>
                </p>
                <p style="color: #a8dadc; text-align: center; margin-top: 1rem;">
                    {t['npk_ratio']}: <strong style="color: #00b09b;">{fert_info.get('npk', 'N/A')}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"### {t['fertilizer_details']}")
            
            detail_cols = st.columns(2)
            
            with detail_cols[0]:
                st.info(f"""
                **{t['description']}**
                {fert_info.get('description', 'N/A')}
                
                **{t['best_for']}**
                {fert_info.get('best_for', 'N/A')}
                """)
            
            with detail_cols[1]:
                st.success(f"""
                **{t['application_tips']}**
                {fert_info.get('application', 'N/A')}
                
                **{t['npk_ratio']}:** {fert_info.get('npk', 'N/A')}
                """)
            
            if show_charts and hasattr(model, 'predict_proba'):
                st.markdown(f"### {t['confidence_dist']}")
                
                prob_df = pd.DataFrame({
                    t['fertilizer']: le_fertilizer.classes_,
                    t['probability']: probabilities * 100
                }).sort_values(t['probability'], ascending=True)
                
                prob_fig = px.bar(
                    prob_df, 
                    x=t['probability'], 
                    y=t['fertilizer'],
                    orientation='h',
                    color=t['probability'],
                    color_continuous_scale=['#1a1a2e', '#00b09b', '#96c93d']
                )
                
                prob_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#a8dadc',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(title=t["confidence_pct"], gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    showlegend=False
                )
                
                st.plotly_chart(prob_fig, width="stretch")
                
        except Exception as e:
            st.error(f"{t['error_prediction']}: {e}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #a8dadc; padding: 2rem;">
    <h4>{t['quick_tips']}</h4>
    <p>
        {t['tip1']}<br>
        {t['tip2']}<br>
        {t['tip3']}<br>
        {t['tip4']}
    </p>
  
</div>
""", unsafe_allow_html=True)
