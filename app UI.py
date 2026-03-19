import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import load_model
import base64
import requests
import folium
import polyline
from streamlit_folium import st_folium

# ==========================
# CONFIG
# ==========================
SEQ_LENGTH = 5
PREDICT_HORIZON = 5  # menit ke depan

MODEL_PATH = r"C:\Users\BI2026\Model Without class weight"
DATA_PATH  = r"C:\Users\BI2026\Dataset\lane_counts.csv"

MODEL_FILE  = r"best_ANN_percentile50.h5"
SCALER_FILE = "scaler.save"
ICON_READ = "Assets/icons8-read-96.png"
ICON_WINDOW = "Assets/icons8-big-data-100.png"
ICON_FILL = "Assets/icons8-data-quality-100.png"
LANE_Avisual= "Assets/LaneADataVisual.png"
LANE_Bvisual= "Assets/LaneBDataVisual.png"
LANE_Cvisual= "Assets/LaneCDataVisual.png"
LANE_Dvisual= "Assets/LaneDDataVisual.png"
LANE_Evisual= "Assets/LaneEDataVisual.png"
LANE_Fvisual= "Assets/LaneFDataVisual.png"
ORS_API_KEY = st.secrets["ORS_API_KEY"]
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

FEATURE_COLS = [
    "Vehicle_Count",
    "Delta",
    "Rolling_Mean_3",
    "Rolling_Mean_5"
]

def img_to_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()
ICON_READ_B64 = img_to_base64(ICON_READ)
ICON_WINDOW_B64 = img_to_base64(ICON_WINDOW)
ICON_FILL_B64 = img_to_base64(ICON_FILL)
LANE_A_B64 = img_to_base64(LANE_Avisual)
LANE_B_B64 = img_to_base64(LANE_Bvisual)
LANE_C_B64 = img_to_base64(LANE_Cvisual)
LANE_D_B64 = img_to_base64(LANE_Dvisual)
LANE_E_B64 = img_to_base64(LANE_Evisual)
LANE_F_B64 = img_to_base64(LANE_Fvisual)
LANE_A_LB64 = img_to_base64("Assets/Lane_A_label_distribution.png")
LANE_B_LB64 = img_to_base64("Assets/Lane_B_label_distribution.png")
LANE_C_LB64 = img_to_base64("Assets/Lane_C_label_distribution.png")
LANE_D_LB64 = img_to_base64("Assets/Lane_D_label_distribution.png")
LANE_E_LB64 = img_to_base64("Assets/Lane_E_label_distribution.png")
LANE_F_LB64 = img_to_base64("Assets/Lane_F_label_distribution.png")

html_input_layer = """
<div class="pipeline-card">
<div class="pipeline-title">Input Layer</div>
<div class="pipeline-desc">

The model receives traffic data from the previous <b>5 minutes</b>.

<div style="margin-top:10px">
Each minute contains four features:
</div>

<div style="margin-top:6px">
• Vehicle Count<br>
• Delta<br>
• Rolling Mean 3<br>
• Rolling Mean 5
</div>

<div style="margin-top:10px">
<b>Input Shape: (5, 4)</b>
</div>

</div>
</div>
"""
html_flatten = """
<div class="pipeline-card">
<div class="pipeline-title">Flatten Layer</div>
<div class="pipeline-desc">

The Flatten layer converts the sequence into a single vector.

<div style="margin-top:10px">
<b>Original Shape</b>
</div>

(5 minutes × 4 features)

<div style="margin-top:10px">
<b>After Flatten</b>
</div>

20 input values

<div style="margin-top:10px">
This allows a standard neural network to process time window data.
</div>

</div>
</div>
"""
html_hidden = """
<div class="pipeline-card">
<div class="pipeline-title">Hidden Layers</div>
<div class="pipeline-desc">

The network uses two fully connected layers.

<div style="margin-top:10px">
<b>Dense(64, ReLU)</b><br>
Learns complex traffic patterns.
</div>

<div style="margin-top:10px">
<b>Dense(32, ReLU)</b><br>
Extracts important features.
</div>

<div style="margin-top:10px">
ReLU activation helps the model learn non-linear relationships.
</div>

</div>
</div>
"""
html_output = """
<div class="pipeline-card">
<div class="pipeline-title">Output Layer</div>

<div class="pipeline-desc">

<b>Dense(1, Sigmoid)</b>

<div style="margin-top:10px">
The final layer produces a probability value between <b>0 and 1</b>.
</div>

<div style="margin-top:10px">
Prediction rule:
</div>

<div style="margin-top:6px">
Probability > 0.5 → HIGH Traffic<br>
Probability <= 0.5 → LOW Traffic
</div>

<div style="margin-top:10px">
Sigmoid activation is used because this is a <b>binary classification problem</b>.
</div>

</div>
</div>
"""
html_training = """
<div class="pipeline-card">
<div class="pipeline-title">Model Training</div>

<div class="pipeline-desc">

<b>Optimizer</b><br>
Adam (learning rate = 0.001)

<div style="margin-top:10px">
<b>Loss Function</b><br>
Binary Crossentropy
</div>

<div style="margin-top:10px">
<b>Evaluation Metric</b><br>
Accuracy
</div>

<div style="margin-top:10px">
Binary crossentropy measures how well the predicted probability 
matches the true traffic label (LOW or HIGH).
</div>

</div>
</div>
"""

# ==========================
# LOAD DATA
# ==========================

def get_route(start, end):

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "coordinates": [
            start,
            end
        ]
    }

    res = requests.post(
        ORS_URL,
        json=body,
        headers=headers
    )

    data = res.json()

    geometry = data["routes"][0]["geometry"]

    coords = polyline.decode(geometry)

    return coords
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Minute_Window"])
    df = df.sort_values(["Lane", "Minute_Window"])
    return df

data = load_data()

# ==========================
# LOAD SCALER & MODEL
# ==========================
@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_PATH, SCALER_FILE))

@st.cache_resource
def load_best_model():
    return load_model(os.path.join(MODEL_PATH, MODEL_FILE))

scaler = load_scaler()
model  = load_best_model()
st.markdown("""
<style>

.pipeline-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    transition: all 0.2s ease;
}

.pipeline-card:hover {
    border-color: #4f46e5;
    transform: translateY(-3px);
}

.pipeline-title{
    color: #111111;
    font-weight: 600;
    font-size: 18px;
    margin-top: 8px;
}

.pipeline-desc{
    color: #333333;
    font-size: 14px;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.2s ease;
}

.feature-card:hover {
    border-color: #2563eb;
    transform: translateY(-2px);
}

.feature-title{
    font-weight:600;
    font-size:16px;
    color:#111111;
}

.feature-desc{
    font-size:13px;
    color:#444444;
}
.pipeline-card p{
    margin:6px 0;
    line-height:1.5;
}

.pipeline-card ul{
    text-align:left;
    margin:8px 0;
    padding-left:18px;
}

.pipeline-card li{
    margin-bottom:4px;
}
.result-card{
background:#ffffff;
border:1px solid #e5e7eb;
border-radius:14px;
padding:20px;
text-align:center;
transition:0.2s;
}

.result-card:hover{
border-color:#4f46e5;
transform:translateY(-3px);
}

.result-title{
font-size:18px;
font-weight:600;
margin-bottom:10px;
color:#111;
}

.result-desc{
font-size:14px;
color:#333;
}

.result-acc{
font-size:22px;
font-weight:700;
color:#2563eb;
margin-top:10px;
}

            .overall-card{
background:#ffffff;
border:1px solid #e5e7eb;
border-radius:16px;
padding:30px;
text-align:center;
margin-top:10px;
}

.overall-title{
font-size:20px;
font-weight:600;
color:#111;
margin-bottom:10px;
}

.overall-model{
font-size:16px;
color:#333;
margin-top:5px;
}

.overall-score{
font-size:34px;
font-weight:700;
color:#2563eb;
margin-top:10px;
}

.overall-sub{
font-size:14px;
color:#555;
margin-top:8px;
}        

</style>
""", unsafe_allow_html=True)
results = {
"Lane_A":0.91,
"Lane_B":0.84,
"Lane_C":0.87,
"Lane_D":0.93,
"Lane_E":0.80,
"Lane_F":0.87
}
# ==========================
# UI
# ==========================
st.title("🚦 Traffic Prediction System")
st.markdown("## ⚙️ Data Processing Pipeline")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="pipeline-card">
        <img src="data:image/png;base64,{ICON_READ_B64}" width="70">
        <div class="pipeline-title">Read Raw Data</div>
        <div class="pipeline-desc">Load vehicle timestamps from toll lanes.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="pipeline-card">
        <img src="data:image/png;base64,{ICON_WINDOW_B64}" width="70">
        <div class="pipeline-title">Minute Windowing</div>
        <div class="pipeline-desc">Aggregate traffic into 1-minute counts.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="pipeline-card">
        <img src="data:image/png;base64,{ICON_FILL_B64}" width="70">
        <div class="pipeline-title">Fill Missing Data</div>
        <div class="pipeline-desc">Missing minutes are filled with 0 traffic.</div>
    </div>
    """, unsafe_allow_html=True)
st.divider()
st.markdown("## 📊 Model Features")

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Vehicle Count</div>
        <div class="feature-desc">
        Number of vehicles passing the lane within one minute.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Delta</div>
        <div class="feature-desc">
        Change in vehicle count compared to the previous minute.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Rolling Mean 3</div>
        <div class="feature-desc">
        Average vehicle count over the last 3 minutes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with f4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Rolling Mean 5</div>
        <div class="feature-desc">
        Average vehicle count over the last 5 minutes.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("## 📈 Data Analytics")
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane A</div>
        <img src="data:image/png;base64,{LANE_A_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane B</div>
        <img src="data:image/png;base64,{LANE_B_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row1_col3:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane C</div>
        <img src="data:image/png;base64,{LANE_C_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)


row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane D</div>
        <img src="data:image/png;base64,{LANE_D_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane E</div>
        <img src="data:image/png;base64,{LANE_E_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row2_col3:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane F</div>
        <img src="data:image/png;base64,{LANE_F_B64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("## 📈 Label Distribution Analytics")

# ===== ROW 1 =====
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane A</div>
        <img src="data:image/png;base64,{LANE_A_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane B</div>
        <img src="data:image/png;base64,{LANE_B_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row1_col3:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane C</div>
        <img src="data:image/png;base64,{LANE_C_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

# ===== SPACING BETWEEN ROWS =====
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

# ===== ROW 2 =====
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane D</div>
        <img src="data:image/png;base64,{LANE_D_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane E</div>
        <img src="data:image/png;base64,{LANE_E_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

with row2_col3:
    st.markdown(f"""
    <div class="pipeline-card">
        <div class="pipeline-title">Lane F</div>
        <img src="data:image/png;base64,{LANE_F_LB64}" style="width:100%; border-radius:10px;">
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("## 📊 Traffic Threshold & Model Input")

percent_steps = [0, 25, 50, 75, 100]

lane_percentiles = (
    data
    .groupby("Lane")["Vehicle_Count"]
    .quantile([p/100 for p in percent_steps])
    .unstack()
)

lane_percentiles.columns = ["P0","P25","P50","P75","P100"]
lane_percentiles = lane_percentiles.astype(int)

# membuat lane jadi kolom agar tabel lebih jelas
lane_percentiles = lane_percentiles.reset_index()

# layout kolom
col1, col2 = st.columns([3,2])

with col1:

    st.markdown("### Vehicle Count Percentiles per Lane")

    st.table(lane_percentiles)

    st.caption(
        "The P50 value represents the median vehicle count used as the threshold "
        "to classify traffic into LOW and HIGH conditions."
    )

with col2:

    st.markdown("""
    <div class="pipeline-card">
        <div class="pipeline-title">Model Input Sequence</div>
        <div class="pipeline-desc">
        The prediction model uses <b>5 previous minutes of traffic data</b> 
        as input features.
        <br><br>
        <b>SEQ_LENGTH = 5</b>
        <br><br>
        This means the model analyzes the traffic pattern from 
        the last five minutes before generating a prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)
st.divider()

st.markdown("## 🧠 Model Architecture (ANN)")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(html_input_layer, unsafe_allow_html=True)

with c2:
    st.markdown(html_flatten, unsafe_allow_html=True)

with c3:
    st.markdown(html_hidden, unsafe_allow_html=True)

c4, c5 = st.columns(2)

with c4:
    st.markdown(html_output, unsafe_allow_html=True)

with c5:
    st.markdown(html_training, unsafe_allow_html=True)
st.divider()

st.markdown("## 📊 Cross-Lane Validation Results")
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane A</div>
    <div class="result-desc">

    Train: Lane B, C, D, E, F<br>
    Test: Lane A<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_A"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane B</div>
    <div class="result-desc">

    Train: Lane A, C, D, E, F<br>
    Test: Lane B<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_B"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane C</div>
    <div class="result-desc">

    Train: Lane A, B, D, E, F<br>
    Test: Lane C<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_C"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)

col4,col5,col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane D</div>
    <div class="result-desc">

    Train: Lane A, B, C, E, F<br>
    Test: Lane D<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_D"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane E</div>
    <div class="result-desc">

    Train: Lane A, B, C, D, F<br>
    Test: Lane E<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_E"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="result-card">
    <div class="result-title">Test Lane F</div>
    <div class="result-desc">

    Train: Lane A, B, C, D, E<br>
    Test: Lane F<br><br>

    X_train: (275,5,4)<br>
    X_test: (55,5,4)

    </div>

    <div class="result-acc">
    Accuracy: {results["Lane_F"]:.2f}
    </div>

    </div>
    """, unsafe_allow_html=True)
st.markdown("### Overall Model Performance")
st.markdown("""
<div class="overall-card">

<div class="overall-title">
Cross-Lane Performance Summary
</div>

<div class="overall-model">
Model: Artificial Neural Network (ANN)
</div>

<div class="overall-score">
0.8697 ± 0.0425
</div>

<div class="overall-sub">
Mean Accuracy ± Standard Deviation across all test lanes
</div>

</div>
""", unsafe_allow_html=True)




st.markdown("## 🗺️ Route Simulation")

start_lat = -6.283046
start_lon = 107.1689459

end_lat = -6.2827078
end_lon = 107.1707595

if "prediction_table" not in st.session_state:
    st.session_state["prediction_table"] = None

if "show_route" not in st.session_state:
    st.session_state["show_route"] = False

if "traffic_condition" not in st.session_state:
    st.session_state["traffic_condition"] = "LOW"




st.markdown(
    """
**System Goal:**  
If you depart at a specific minute, the system predicts whether the traffic you will encounter is likely to be HIGH or LOW.

**Output:** Probability of HIGH traffic (threshold = 0.5)
"""
)

minute_selected = st.selectbox(
    "🕒 Select Time",
    sorted(data["Minute_Window"].unique())
)

if st.button("🔍 Predict Traffic Condition"):

    st.subheader("🚘 Prediction Result")

    results = []

    for lane in sorted(data["Lane"].unique()):

        lane_df = data[data["Lane"] == lane].reset_index(drop=True)

        match = lane_df[lane_df["Minute_Window"] == minute_selected]

        if match.empty:
            continue

        idx = match.index[0]

        if idx < SEQ_LENGTH:
            continue

        probs = []

        for step in range(PREDICT_HORIZON):

            current_idx = idx + step

            if current_idx >= len(lane_df):
                break

            history_df = lane_df.iloc[
                current_idx - SEQ_LENGTH : current_idx
            ].copy()

            history_df["Delta"] = history_df["Vehicle_Count"].diff().fillna(0)

            history_df["Rolling_Mean_3"] = (
                history_df["Vehicle_Count"]
                .rolling(3, min_periods=1)
                .mean()
            )

            history_df["Rolling_Mean_5"] = (
                history_df["Vehicle_Count"]
                .rolling(5, min_periods=1)
                .mean()
            )

            scaled = scaler.transform(history_df[FEATURE_COLS])

            X_input = np.expand_dims(scaled, axis=0)

            prob_high = model.predict(X_input, verbose=0)[0][0]

            probs.append(prob_high)

        if len(probs) == 0:
            continue

        avg_prob = float(np.mean(probs))

        if avg_prob >= 0.5:
            label = "HIGH"
        else:
            label = "LOW"

        results.append({
            "Lane": lane,
            "Prediction": label,
            "Probability": avg_prob
        })


    results_df = pd.DataFrame(results)

    # simpan ke session_state
    st.session_state["prediction_table"] = results_df


    if results_df["Prediction"].value_counts().get("HIGH",0) > \
       results_df["Prediction"].value_counts().get("LOW",0):

        st.session_state["traffic_condition"] = "HIGH"
    else:
        st.session_state["traffic_condition"] = "LOW"

    st.session_state["show_route"] = True
if st.session_state.get("show_route"):

    start = [start_lon, start_lat]
    end = [end_lon, end_lat]
    traffic = st.session_state.get("traffic_condition", "LOW")

    if traffic == "HIGH":
        route_color = "red"
    else:
        route_color = "green"

    route = get_route(start, end)  # route mengikuti jalan

    m = folium.Map(
        location=[start_lat, start_lon],
        zoom_start=16,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False,
        touchZoom=False
    )

    # gambar route
    folium.PolyLine(
        route,
        color=route_color,
        weight=5
    ).add_to(m)

    lanes = 6

    step = len(route) // lanes

    for i in range(lanes):

        point = route[i * step]

        folium.Marker(
            point,
            tooltip=f"Lane {i+1}",
            icon=folium.Icon(
                color=route_color,
                icon="info-sign"
            )
        ).add_to(m)

    st_folium(m, width=900, height=500)

if st.session_state["prediction_table"] is not None:

    df = st.session_state["prediction_table"]

    styled_df = df.style.apply(
        lambda row: [
            "background-color:red;color:white"
            if row.Prediction == "HIGH" and col == "Probability"
            else "background-color:green;color:white"
            if row.Prediction == "LOW" and col == "Probability"
            else ""
            for col in df.columns
        ],
        axis=1
    ).format({
        "Probability": "{:.2%}"
    })

    st.subheader("🚘 Prediction Result")
    st.dataframe(styled_df, use_container_width=True)