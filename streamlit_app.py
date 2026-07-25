import streamlit as st
import pandas as pd

st.title("AM1 Project Dashboard")

df = pd.read_csv("data/Level 6 AM1 Dataset.csv")

st.write("### Dataset preview")
st.dataframe(df)
