import logging
import os
import asyncio
import uuid
import gradio as gr
from analysis_agent import agent_executor

def same_auth(username, password):
    return username == password

def main():
    async def gr_interface(message, history, request: gr.Request):
        logging.info(message)
        logging.info(f"request.client.host {request.client.host}")
        logging.info(f"Client port: {request.client.port}")
        logging.info(f"Request method: {request.method}")
        logging.info(f"Request headers: {request.headers}")
        logging.info(f"Request URL: {request.url}")
        logging.info(f"Request query parameters: {request.query_params}")
        logging.info(f"Request cookies: {request.cookies}")
        logging.info(f"Request body: {request.body}")
        logging.info(f"Client information: {request.client}")
        request_message = message["text"] if isinstance(message, dict) else message
        config = {"configurable": {"thread_id": request.username}}
        logging.info(f"Request message: {request_message}")
        logging.info(f"Request config: {config}")

        response = await agent_executor.ainvoke(
            {"messages": [("user", request_message)]}, config=config
        )
        logging.info(response['messages'][-1].content)
        return response['messages'][-1].content

    logging.basicConfig(
        level=logging.INFO,
        format="\033[93m%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s -\033[0m %(message)s",
    )

    conversation_id = str(uuid.uuid4())

    gr.ChatInterface(
        gr_interface,
        type="messages",
        multimodal=True,
        title="Veri Analiz Asistanı",
        description="Veri Analiz Asistanı",
        theme="glass",
        examples=[
            "Analiz yapabileceğim veriler nelerdir?",
            "Tekil sınıfları listele.",
            "Toplam kaç adet istek gelmiş?",
        ],
    ).launch(auth=same_auth, pwa=True, share=True)
    #     multimodal=True,
    #     chatbot=gr.Chatbot(height=800),
    #     textbox=gr.Textbox(placeholder="Enter your question !", container=False, scale=7),
    #     title="BILIN SSS",
    #     description="Bilin Sıkça Sorulan Sorular",
    #     theme="soft",
    #     examples=["Katalog dışı eğitim talebi kapatılabilir mi?", "Döviz cinsi tanımlamayı nasıl yapabilirim?", "Yıllık izinleri nasıl listeleyebilirim?"],
    #     cache_examples=False,
    # ).launch()


# demo = gr.ChatInterface(random_response, type="messages", autofocus=True)

if __name__ == "__main__":
    main()
