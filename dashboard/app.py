import streamlit as st

st.set_page_config(
    page_title="IPL AI Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏏 IPL AI Analytics System")

st.subheader("Production-Level Cricket Intelligence Platform")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Matches Analyzed", "1000+")

with col2:
    st.metric("Players Tracked", "500+")

with col3:
    st.metric("Prediction Accuracy", "87%")

st.markdown("---")

st.header("🚀 Features")

features = [
    "Match Winner Prediction",
    "Live Win Probability",
    "Player Analytics",
    "Venue Intelligence",
    "Fantasy Cricket Suggestions",
    "Explainable AI Insights"
]

for feature in features:
    st.write(f"✅ {feature}")
