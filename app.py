import streamlit as st
from dotenv import load_dotenv
import os
from stock_picker import stock_picker_crew


load_dotenv()

os.makedirs("output", exist_ok=True)

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

st.title("Stock Picker Agent")
st.write("AI-powered investment research using 3 specialized agents")

sector = st.selectbox("Choose a sector:", ["technology", "healthcare", "energy", "finance"])

if st.button("Run Analysis"):
    with st.spinner("Agents are researching... (this takes 2-5 minutes)"):

        
        result = stock_picker_crew.kickoff(inputs={"sector": sector})
    st.markdown(result.raw)
    st.download_button("Download Report", result.raw, file_name="stock_recommendations.md")