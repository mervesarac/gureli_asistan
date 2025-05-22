import logging
import os
import asyncio
import json
import uuid
import gradio as gr
from eguven_analysis_agent import agent_executor
from faker import Faker
from faker_music import MusicProvider

def get_profile():
    fake = Faker('tr_TR')
    fake.add_provider(MusicProvider)
    profile = fake.profile()
    profile["music"] = [fake.music_genre() for _ in range(5)]
    profile = {
        "name": "MERVE SARAÇ",
        "company": "Mersis Bilgi Teknolojileri Danışmanlık Ltd.",
        "email": "merve@mersis.com.tr",
        "phone_number": "532 236 42 32",
    }
    return json.dumps(profile, default=str, ensure_ascii=False)

def same_auth(username, password):
    return username == password

def main():
    async def gr_interface(message, history, request: gr.Request):
        logging.info(message)
        request_message = message["text"] if isinstance(message, dict) else message
        config = {"configurable": {"thread_id": request.username}, "recursion_limit": 25}
        logging.info(f"Request message: {request_message}")
        logging.info(f"Request config: {config}")

        if len(history) == 0:
            messages = [("user", f"Merhaba! Profil bilgilerimi paylaşıyorum: {get_profile()}")]
            messages.append(("user", request_message))
        else:
            messages = [("user", request_message)]
        response = await agent_executor.ainvoke(
            {"messages": messages}, config=config, 
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
        title="AGENT AI ile VERİ ANALİTİĞİ",
        description="Veri Analiz Asistanı",
        theme="glass",
        examples=[
            "Analiz edebileceğim tabloları listele.",
            "Attrition tablosunda hangi bilgiler var?",
            "Attrition tablosunda kaç çalışan var?",
            "Yaşa göre maaş ortalamalarını hesapla.",
        ],
        autofocus=True,
    ).launch(pwa=True, share=True) # auth=same_auth, 
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
