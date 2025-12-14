# iris_app.py
# Iris Flower Species Prediction App (Deployment Ready)

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("iris-species.csv")

    # Encode target variable
    df["Label"] = df["Species"].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    })
    return df


iris_df = load_data()

# --------------------------------------------------
# Prepare Data
# --------------------------------------------------
X = iris_df[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]
y = iris_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# --------------------------------------------------
# Train Models (Cached)
# --------------------------------------------------
@st.cache_resource
def train_models():
    svc = SVC(kernel="linear")
    svc.fit(X_train, y_train)

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    rf.fit(X_train, y_train)

    return svc, log_reg, rf


svc_model, log_reg_model, rf_model = train_models()

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_species(model, sl, sw, pl, pw):
    label = model.predict(
        np.array([[sl, sw, pl, pw]])
    )[0]

    species_map = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    }
    return species_map[label]

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#e63946;'>"
    "🌸 Iris Flower Species Prediction App"
    "</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("🌼 Model Configuration")

# --------------------------------------------------
# Sidebar Sliders
# --------------------------------------------------
s_len = st.sidebar.slider(
    "Sepal Length (cm)",
    float(X["SepalLengthCm"].min()),
    float(X["SepalLengthCm"].max())
)

s_wid = st.sidebar.slider(
    "Sepal Width (cm)",
    float(X["SepalWidthCm"].min()),
    float(X["SepalWidthCm"].max())
)

p_len = st.sidebar.slider(
    "Petal Length (cm)",
    float(X["PetalLengthCm"].min()),
    float(X["PetalLengthCm"].max())
)

p_wid = st.sidebar.slider(
    "Petal Width (cm)",
    float(X["PetalWidthCm"].min()),
    float(X["PetalWidthCm"].max())
)

# --------------------------------------------------
# Classifier Selection
# --------------------------------------------------
classifier = st.sidebar.selectbox(
    "Select Classifier",
    (
        "Support Vector Machine",
        "Logistic Regression",
        "Random Forest Classifier"
    )
)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.sidebar.button("Predict"):
    if classifier == "Support Vector Machine":
        model = svc_model
    elif classifier == "Logistic Regression":
        model = log_reg_model
    else:
        model = rf_model

    species = predict_species(
        model, s_len, s_wid, p_len, p_wid
    )
    accuracy = model.score(X_test, y_test)

    st.success(f"🌸 Predicted Species: **{species}**")
    st.info(f"📊 Model Accuracy: **{accuracy:.2f}**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("Built this project during Whitehat course, update fixed")
