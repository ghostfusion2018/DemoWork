from langchain_experimental.agents import create_csv_agent
#from langchain.llms import OpenAI
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import streamlit as st

st.set_page_config(page_title="Query your CSV", page_icon=":chart_with_upwards_trend:")
st.header("Query your CSV :chart_with_upwards_trend:")

llm = Ollama(base_url="http://localhost:11434", model="starling-lm")
csv_file = st.file_uploader("Upload a CSV file", type="csv")
if csv_file is not None:
    agent = create_csv_agent(llm, csv_file, verbose=True)

    user_question = st.text_input("Ask a question about your CSV: ")

    if user_question is not None and user_question != "":
        with st.spinner(text="In progress..."):
            st.write(agent.run(user_question))

    # agent = create_csv_agent(
    # OpenAI(temperature=0), csv_file, verbose=True)




