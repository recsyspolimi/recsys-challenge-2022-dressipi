import telegram
import requests


def telegram_bot_sendtext(bot_message):
    bot_message = telegram.utils.helpers.escape_markdown(bot_message)
    bot_token = '5131758164:AAGUx6-znhBvBCAOHGZ5Qmcm28ji6KMuBPI'
    bot_chatID = '257525257'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    bot_chatID = '67336969'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    bot_chatID = '35737327'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()


def telegram_bot_sendfile(file, file_name):
    bot_file = {'document': (file_name, open(file, 'rb'))}
    bot_token = '5131758164:AAGUx6-znhBvBCAOHGZ5Qmcm28ji6KMuBPI'
    bot_chatID = '257525257'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID

    response = requests.post(send_text, files=bot_file)

    bot_file = {'document': (file_name, open(file, 'rb'))}
    bot_chatID = '67336969'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID

    response = requests.post(send_text, files=bot_file)

    bot_file = {'document': (file_name, open(file, 'rb'))}
    bot_chatID = '35737327'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendDocument?chat_id=' + bot_chatID

    response = requests.post(send_text, files=bot_file)

    return response.json()
