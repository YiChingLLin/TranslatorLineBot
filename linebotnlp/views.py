from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden 
from django.views.decorators.csrf import csrf_exempt 
from django.conf import settings

from linebot import LineBotApi, WebhookParser 
from linebot.exceptions import InvalidSignatureError, LineBotApiError 
from linebot.models import MessageEvent, TextSendMessage

import torch
import torchvision
import transformers
import sentencepiece
import sacremoses
from transformers import M2M100ForConditionalGeneration
from .tokenization_small100 import SMALL100Tokenizer

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

# 使用SMALL-100 Model
model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

def translate(lang, msg): # 將欲翻譯訊息傳入模型後得到翻譯結果
    tokenizer.tgt_lang = lang # 設定欲翻譯語言
    encoded_zh = tokenizer(msg, return_tensors="pt")
    generated_tokens = model.generate(**encoded_zh, max_new_tokens = 1000)
    trans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return trans[0]

def set_language(msg): # 辨識使用者欲翻譯的語言代碼
    if msg == '英':
        language = 'en' # 英文
    elif msg == '西':
        language = 'es' # 西班牙文
    elif msg == '日':
        language = 'ja' # 日文
    elif msg == '韓':
        language = 'ko' # 韓文
    elif msg == '法':
        language = 'fr' # 法文
    else: # 不支援的語言種類
        language = 'error' # 錯誤
    
    return language

@csrf_exempt
def callback(request):
    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode ('utf-8')
        
        try:
            events = parser.parse(body, signature)
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()

        for event in events:
            if isinstance(event, MessageEvent) :
                lang = event.message.text[:1] # 欲翻譯的語言, ex:使用者輸入 「英 您好」, 取「英」
                msg = event.message.text[2:]  # 欲翻譯的句子, ex:使用者輸入 「英 您好」, 取「您好」
                language = set_language(lang) # 回傳欲翻譯語言的代碼

                if language != 'error': # 支援翻譯的語言
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text = translate(language, msg)) # 回傳翻譯結果
                    ) 
                else: # 不支援翻譯的語言或未輸入欲翻譯的語言
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text = '請輸入欲翻譯語言') # 回傳錯誤訊息
                    )
        return HttpResponse()
    else:
        return HttpResponseBadRequest()