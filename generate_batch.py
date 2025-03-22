import pandas as pd

from pydantic import BaseModel
from typing import List, Optional

{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

class Format(BaseModel):
   type: str = "json_object"

class Message(BaseModel):
   role: str
   content: str

class Body(BaseModel):
   model: str = "gpt-4o"
   response_format: Format
   messages: List[Message]
   max_tokens: int = 2000

class Request(BaseModel):
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: Body

def replace_turkish_chars(text):
  """
  Replaces Turkish characters with their closest English equivalents.

  Args:
      text: The text to be converted.

  Returns:
      The converted text with Turkish characters replaced.
  """
  char_map = {
    'i':'ı',
    'I':'İ',
    'g':'ğ',
    'G':'Ğ',
    'o':'ö',
    'O':'Ö',
    's':'ş',
    'S':'Ş',
    'c':'ç',
    'C':'Ç',
    'u':'ü',
    'U':'Ü'
  }
  n = []
  for c in text:
    try:
      c.encode("latin5")
      n.append(c)
    except:
      # print(''.join(n))
      p = n.pop()
      n.append(char_map.get(p, p))
      # print(''.join(n))

  return ''.join(n)

def remove_blanks(text):
    return text.replace(" ", "")

complain_class_list = ['YENİLEME BAŞVURUSU',
 'ZAMAN DAMGASI',
 'SERTİFİKA AKTARIM',
 'TEKNİK DESTEK',
 'Kurulum Talebi',
 'WEB',
 'Sertifika İptal & Askı',
 'PIN-PUK BİLGİSİ - ŞİFRE BLOKESİ (Token )',
 'DİĞER',
 'KURULUM BİLGİSİ-PROGRAM YÜKLEME',
 'KARGO TAKİP BİLGİSİ',
 'KURUMSAL ÜRÜN SATINALMA-DESTEK TALEPLERİ',
 'EBİMZA',
 'Müşteri Hizmetlerine Ulaşamama Sorunu',
 'PTT Hizmetinde Kimlik Doğrulama Sorunu Ve Uzun Bekleme Süresi',
 'FATURA–MUHASEBE'
 'PERSONEL HAKKINDA GÖRÜŞLER',
 'YENİ BAŞVURU',
 'E-Güven Müşteri Hizmetlerinde Uzun Bekleme Süresi Sorunu',
 'PTT SÜREÇLERİ',
 'E-güven Müşteri Hizmetlerinde Uzun Bekleme Süresi']

tickets = pd.read_excel('crm_subat.xlsx', parse_dates=['Datetime'])
with open("tickets_output.jsonl", "w") as output_file:
    for index, row in tickets.iterrows():
        system_message = Message(
            role="system",
            content=f"classify digital signature services customer complain as one of the following list {complain_class_list}. The response should be in json format and include row number, complain and class"
        )
        user_message = Message(
            role="user",
            content=row["Requirement"]
        )
        format = Format()
        body = Body(
            messages=[system_message, user_message],
            response_format=format
        )
        request = Request(
            custom_id=f"request-{index + 1}",
            body=body
        )
        output_file.write(f"{request.model_dump_json()}\n")
        if index > 4:
            break
