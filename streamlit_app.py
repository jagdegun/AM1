import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

st.title("AM1 Project Dashboard")

# Load raw data (used for dropdown options, so all locations show up)
df_raw = pd.read_csv("data/Level 6 AM1 Dataset.csv")

st.write("### Dataset preview")
st.dataframe(df_raw.head())

# Cleaned data (used for training - drops incomplete rows)
features = ['Location', 'Gender', 'Age']
df = df_raw.dropna(subset=features + ['Advertisement Type'])

X = df[features]
y = df['Advertisement Type']
X_encoded = pd.get_dummies(X, drop_first=True)

# --- Evaluation model (train/test split, just to report accuracy) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
scaler_eval = StandardScaler()
X_train_scaled = scaler_eval.fit_transform(X_train)
X_test_scaled = scaler_eval.transform(X_test)

eval_model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
eval_model.fit(X_train_scaled, y_train)
y_pred = eval_model.predict(X_test_scaled)

st.write("### Model performance (evaluated on held-out test data)")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# --- Final model, trained on the WHOLE cleaned dataset (used for predictions) ---
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_encoded)

final_model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')
final_model.fit(X_full_scaled, y)

# --- Interactive prediction ---
st.write("### Try a prediction")

# Pull dropdown options from the RAW data, so all locations (e.g. Birmingham) appear
location_input = st.selectbox("Location", sorted(df_raw['Location'].dropna().unique()))
gender_input = st.selectbox("Gender", sorted(df_raw['Gender'].dropna().unique()))
age_input = st.selectbox("Age", sorted(df_raw['Age'].dropna().unique()))

if st.button("Predict Advertisement Type"):
    new_data = pd.DataFrame([{
        'Location': location_input,
        'Gender': gender_input,
        'Age': age_input
    }])

    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    new_data_scaled = scaler_full.transform(new_data_encoded)

    prediction = final_model.predict(new_data_scaled)
    st.success(f"Predicted Advertisement Type: **{prediction[0]}**")
