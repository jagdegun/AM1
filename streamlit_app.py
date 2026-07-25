import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

st.title("AM1 Project Dashboard")

# Load and clean data
df = pd.read_csv("data/Level 6 AM1 Dataset.csv")
df = df.dropna()
df['Advertisement Release Time '] = df['Advertisement Release Time '].astype(str)

st.write("### Dataset preview")
st.dataframe(df.head())

# Features and target
features = ['Consultation', 'Location', 'Gender', 'Age', 'Advertisement Release Time ']
X = df[features]
y = df['Advertisement Type']
X_encoded = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    multi_class='multinomial', max_iter=1000, solver='lbfgs', class_weight='balanced'
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Show performance
st.write("### Model performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# --- Interactive prediction ---
st.write("### Try a prediction")

consultation_input = st.selectbox("Consultation", df['Consultation'].unique())
location_input = st.selectbox("Location", df['Location'].unique())
gender_input = st.selectbox("Gender", df['Gender'].unique())
age_input = st.selectbox("Age", df['Age'].unique())
release_time_input = st.selectbox("Advertisement Release Time", df['Advertisement Release Time '].unique())

if st.button("Predict Advertisement Type"):
    new_data = pd.DataFrame([{
        'Consultation': consultation_input,
        'Location': location_input,
        'Gender': gender_input,
        'Age': age_input,
        'Advertisement Release Time ': release_time_input
    }])

    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data_encoded)

    prediction = model.predict(new_data_scaled)
    st.success(f"Predicted Advertisement Type: **{prediction[0]}**")
