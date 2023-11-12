import telebot
import time
import os
from dotenv import load_dotenv
from threading import Thread
from virtual_db import VirtualDB
from inotify.adapters import Inotify

load_dotenv()

bot = telebot.TeleBot(
    os.environ['TG_KEY'], parse_mode=None
)  # You can set parse_mode by default. HTML or MARKDOWN

db = VirtualDB()


# когда вышли новые отчёты -- отправляем их
def send_thread():
    if not os.path.exists('common_result.csv'):
        os.system('touch common_result.csv')
    inotify = Inotify()
    inotify.add_watch('common_result.csv')

    for event in inotify.event_gen(yield_nones=False):
        (_, type_names, path, filename) = event
        if 'IN_CLOSE_WRITE' in type_names:
            for id in db.select_all('id'):
                bot.send_document(
                    id,
                    open('common_result.csv', 'rb'),
                    caption='Файл с данными за прошлый день:',
                )
                bot.send_document(
                    id,
                    open('result.csv', 'rb'),
                    caption='Сводка по участкам за прошлый день:',
                )


thread = Thread(target=send_thread, daemon=True)
thread.start()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(
        message,
        "Приветствую! Я -- демонстрационный бот для кейса KOBLiK на Цифровом Прорыве!\n\nЧтобы подписаться на обновления в конце каждого рабочего дня нажмите /subscribe",
    )


@bot.message_handler(commands=['subscribe'])
def subscribe(message):
    chat_id = message.chat.id
    db.insert({'id': chat_id})
    bot.reply_to(
        message,
        "Вы успешно подписаны на обновления :)",
    )


bot.infinity_polling()
