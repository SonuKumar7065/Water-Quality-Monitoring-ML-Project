import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# ================================
# Page Config
# ================================
st.set_page_config(page_title="💧 Water Quality Monitoring", layout="wide")
st.title("💧 Water Quality Monitoring using Machine Learning")
st.markdown("An interactive app to analyze water quality and predict **potability** using ML models.")

# ================================
# Sidebar - Controls
# ================================
st.sidebar.header("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

# ================================
# Load Dataset
# ================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset uploaded successfully!")
else:
    st.sidebar.info("Using default dataset (water_potability.csv)")
    df = pd.read_csv("water_potability.csv")

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

if "Potability" not in df.columns:
    st.error("❌ Target column 'Potability' not found in dataset!")
    st.stop()

# ================================
# Tabs
# ================================
tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset", "📊 EDA", "🤖 Model Training", "🧪 Prediction"])

# ================================
# Tab 1: Dataset
# ================================
with tab1:
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Summary Statistics**")
    st.dataframe(df.describe())

# ================================
# Tab 2: Exploratory Data Analysis
# ================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Potability Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(x="Potability", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

    with col2:
        st.write("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ================================
# Tab 3: Model Training
# ================================
with tab3:
    st.subheader("🤖 Model Training & Evaluation")

    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=random_state, stratify=y
    )

    # Define models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        )
    }

    if xgb_available:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=random_state, n_estimators=300
            ))
        ])

    results = []
    trained_models = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": report["weighted avg"]["f1-score"],
            "ROC-AUC": roc_auc
        })
        trained_models[name] = m

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    st.write("### 📊 Model Comparison")
    st.dataframe(results_df)

    # Best model
    best_model_name = results_df.iloc[0]["Model"]
    st.write(f"✅ Best Model: **{best_model_name}**")
    best_model = trained_models[best_model_name]

    # Confusion Matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{best_model_name} - Confusion Matrix")
    st.pyplot(fig)

    # ROC Curve
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax.plot([0,1],[0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{best_model_name} - ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ================================
# Tab 4: Prediction
# ================================
with tab4:
    st.subheader("🧪 Try Your Own Input")

    feature_names = [col for col in df.columns if col != "Potability"]
    user_inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            user_inputs[feature] = st.number_input(
                f"{feature}", value=float(df[feature].median())
            )

    if st.button("Predict Potability"):
        input_data = np.array([[user_inputs[f] for f in feature_names]])

        pred = best_model.predict(input_data)
        st.success("💧 Potable" if pred[0] == 1 else "⚠️ Not Potable")

        # Show model confidence
        prob = best_model.predict_proba(input_data)[0][1]
        st.write(f"**Confidence:** {prob:.2%}")
