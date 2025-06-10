import streamlit as st
import pandas as pd
import altair as alt
import os

LOG_FILE = "emotion_log.csv"

# Set Streamlit config
st.set_page_config(page_title="Face2Serve Dashboard", layout="wide")

st.title("📊 Face2Serve Emotion Dashboard")

# Load emotion data
if not os.path.exists(LOG_FILE):
    st.warning("⚠️ No data found. Please run camera_satisfaction_detect.py first.")
    st.stop()

df = pd.read_csv(LOG_FILE)
df.columns = ["Time", "Emotion", "Satisfaction"]  # Ensure column names

if df.empty:
    st.warning("⚠️ Log file is empty.")
    st.stop()

# Parse timestamp column
df["Time"] = pd.to_datetime(df["Time"])

# Compute stats
satisfied = df[df["Satisfaction"] == "Satisfied"]
unsatisfied = df[df["Satisfaction"] == "Unsatisfied"]
satisfaction_rate = (len(satisfied) / len(df)) * 100

# Display stats
st.metric("😊 Satisfaction Rate", f"{satisfaction_rate:.2f} %")
st.write("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    emotion_bar = alt.Chart(df).mark_bar().encode(
        x=alt.X("Emotion:N", sort="-y"),
        y="count()",
        color="Emotion:N"
    ).properties(title="Emotion Frequency")
    st.altair_chart(emotion_bar, use_container_width=True)

with col2:
    satisfaction_bar = alt.Chart(df).mark_bar().encode(
        x=alt.X("Satisfaction:N", sort=["Satisfied", "Unsatisfied"]),
        y="count()",
        color="Satisfaction:N"
    ).properties(title="Satisfaction Breakdown")
    st.altair_chart(satisfaction_bar, use_container_width=True)

# Trend chart
trend_chart = alt.Chart(df).mark_line(point=True).encode(
    x="Time:T",
    y="count()",
    color="Emotion:N"
).properties(title="Emotion Trend Over Time")
st.altair_chart(trend_chart, use_container_width=True)

# Table
st.write("### 🧾 Last 50 Detections")
st.dataframe(df.tail(50), use_container_width=True)
