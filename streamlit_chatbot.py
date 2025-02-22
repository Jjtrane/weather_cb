import streamlit as st

from langchain import HuggingFaceHub
from huggingface_hub import login
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain
from langchain.prompts import ChatPromptTemplate

hf_token = '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' # sensitive info - would normally not push to git, and would for sure not show in a true release

login(hf_token)


def chatbot():
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        huggingfacehub_api_token=hf_token
    )
    llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'

    chain = APIChain.from_llm_and_api_docs(
        llm,
        api_docs=open_meteo_docs.OPEN_METEO_DOCS,
        limit_to_domains=["https://api.open-meteo.com/"],
        verbose=True
    )

    return chain

questions = [
        'How much longer is daytime (sunrise to sunset) today than the shortest day of the year in my location?',
        'Is the temperature this January higher than last year?',
        'It feels windy today in Copenhagen. Is this common for this time of year?',
        'Is tomorrow going to be cold?'
    ]
llm_chain = chatbot()

llm_chain.run(questions[3])

# Wanted to set up the chatbot in streamlit. Got too short on time..

# prompt = st.chat_input("Write your question here...")
#
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! How may I assist you?"}]
#
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# with st.chat_message("user"):
#     st.markdown(prompt)
#     response = llm_chain.run(prompt)
# with st.chat_message("assistant"):
#     st.markdown(response)
# st.session_state.messages.append({"role": "assistant", "content": response})