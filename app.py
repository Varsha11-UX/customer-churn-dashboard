import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Churn Analyzer", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("*An end-to-end ML pipeline: EDA → Model Training → Live Prediction*")
st.markdown("---")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telecom-Customer-Churn.csv"
    try:
        df = pd.read_csv(url)
    except:
        # Fallback synthetic data if URL fails
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'tenure': np.random.randint(1, 72, n),
            'MonthlyCharges': np.random.uniform(18, 120, n),
            'TotalCharges': np.random.uniform(18, 8000, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n),
            'PhoneService': np.random.choice(['Yes', 'No'], n),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
            'Churn': np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73])
        })
    return df

df = load_data()

# Preprocessing
def preprocess(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

df_clean = preprocess(df)

# Sidebar
st.sidebar.header("⚙️ Navigation")
section = st.sidebar.radio("Go to", ["📈 EDA", "🤖 Model Performance", "🔮 Live Prediction"])

# ─── EDA ───
if section == "📈 EDA":
    st.header("📈 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df['Churn'].value_counts(normalize=True).get('Yes', df_clean['Churn'].mean())) 
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{churn_rate:.1%}" if isinstance(churn_rate, float) else "26.5%")
    col3.metric("Features", df.shape[1] - 1)
    col4.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_counts = df_clean['Churn'].value_counts()
        ax.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
               colors=['#4CAF50', '#F44336'], startangle=90)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df_clean, palette=['#4CAF50', '#F44336'], ax=ax)
        ax.set_xticklabels(['No Churn', 'Churn'])
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    corr = df_clean.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df_clean, x='tenure', hue='Churn', bins=30, palette=['#4CAF50', '#F44336'], ax=ax)
    st.pyplot(fig)

# ─── MODEL ───
elif section == "🤖 Model Performance":
    st.header("🤖 Model Training & Evaluation")

    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training Random Forest model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{report['accuracy']:.2%}")
    col2.metric("Precision", f"{report['1']['precision']:.2%}")
    col3.metric("Recall", f"{report['1']['recall']:.2%}")
    col4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.2%}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 Feature Importances")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
        fig, ax = plt.subplots(figsize=(5, 4))
        feat_imp.plot(kind='barh', color='steelblue', ax=ax)
        ax.invert_yaxis()
        st.pyplot(fig)

    st.session_state['model'] = model
    st.session_state['features'] = X.columns.tolist()

# ─── PREDICTION ───
elif section == "🔮 Live Prediction":
    st.header("🔮 Live Churn Prediction")
    st.markdown("Enter customer details to predict churn probability.")

    # Train model silently
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)", 18.0, 8000.0, float(tenure * monthly_charges))
        senior = st.selectbox("Senior Citizen", [0, 1])

    with col2:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        partner = st.selectbox("Partner", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        phone_service = st.selectbox("Phone Service", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col3:
        internet = st.selectbox("Internet Service", [0, 1, 2], format_func=lambda x: ["DSL", "Fiber optic", "No"][x])
        contract = st.selectbox("Contract", [0, 1, 2], format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
        payment = st.selectbox("Payment Method", [0, 1, 2, 3],
                               format_func=lambda x: ["Bank transfer", "Credit card", "Electronic check", "Mailed check"][x])

    if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
        input_data = pd.DataFrame([[gender, senior, partner, dependents, phone_service,
                                    internet, contract, payment, tenure, monthly_charges, total_charges]],
                                  columns=X.columns[:11] if len(X.columns) >= 11 else X.columns)

        # Align columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[X.columns]

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ **High Churn Risk** — Probability: {prob:.1%}")
            st.markdown("**Recommended Actions:** Offer loyalty discounts, upgrade contract, or assign retention team.")
        else:
            st.success(f"✅ **Low Churn Risk** — Probability: {prob:.1%}")
            st.markdown("**Status:** Customer is likely to stay. Consider upsell opportunities.")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['Churn Risk'], [prob], color='#F44336' if prob > 0.5 else '#4CAF50', height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

st.markdown("---")
st.markdown("*Built with Python · Scikit-learn · Streamlit | Varsha T —  Project*")
