import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Test App", page_icon="✅")

st.title("🧪 Test App - Date Input Only")

# Ultra simple date input
today = datetime.now().date()
date_input = st.date_input("Select Date", today)

st.write(f"Selected date: {date_input}")
st.write(f"Date type: {type(date_input)}")
st.success("✅ If you see this, date input works!")