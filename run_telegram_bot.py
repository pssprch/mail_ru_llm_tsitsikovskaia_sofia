import telebot
from model_wrapper import ModelWrapper
from telebot import types

TOKEN = "..."
bot = telebot.TeleBot(TOKEN)

model_wrapper = ModelWrapper()


@bot.message_handler(commands=['help'])
def help(message):
    help_message = """Доступны следующие команды:
/start Старт
/model Выбор модели
/checkmodel Посмотреть, как модель сейчас загружена
/generate Сгенерировать текст (можно использовать без введения команды)
"""
    bot.send_message(message.from_user.id, help_message)


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.from_user.id, "Привет! Для знакомства с доступными командами введите команду /help")


@bot.message_handler(commands=['model'])
def model(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("StatLM")
    btn2 = types.KeyboardButton("GPT")
    btn3 = types.KeyboardButton("Llama")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "Выберите модель для генерации", reply_markup=markup)


@bot.message_handler(commands=['checkmodel'])
def checkmodel(message):
    bot.send_message(message.from_user.id, f"Текущая модель: {str(model_wrapper.current_model_name)}")


@bot.message_handler(commands=['generate'])
def generate(message):
    bot.send_message(message.from_user.id,
                     "Введите текст (вопрос, на который нужно ответить, либо текст, который нужно продолжить)")


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(f'<{message.text}>')
    if message.text in ['StatLM', 'GPT', 'Llama']:
        print(f'@{message.text}@')
        status, result = model_wrapper.load(message.text, test_inference=True)
        if status:
            bot.send_message(message.from_user.id, "Подгружено")
        else:
            bot.send_message(message.from_user.id, f"Проблемы с загрузкой модели, ниже описаны ошибки.\n{result}")
    else:
        status, result = model_wrapper.generate(message.text)
        if status:
            bot.send_message(message.from_user.id, result)
        else:
            bot.send_message(message.from_user.id, f"Проблемы с генерацией, ниже описаны ошибки.\n{result}")


bot.polling(none_stop=True, interval=0)
