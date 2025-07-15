import asyncio

from requests import session
import streamlit as st
import random
import time
from analysis_agent import agent_executor


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# ,
async def main():
    st.title("Simple chat")

    if "config" not in st.session_state:
        st.session_state.config = {
            "configurable": {"thread_id": "default_thread"},
            "recursion_limit": 25,
        }
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            agent_response = await agent_executor.ainvoke(
            {"messages": st.session_state.messages}, config=st.session_state.config)
            response = st.markdown(agent_response['messages'][-1].content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": agent_response['messages'][-1].content})

if __name__ == "__main__":
    asyncio.run(main())
