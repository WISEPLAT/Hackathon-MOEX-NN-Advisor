"""
    Код торгового советника, в онлайн режиме отображающего предсказания по акциям из индекса

    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

# pip install flask

import os
from flask import Flask, request
from flask import render_template


app = Flask(__name__,
            static_url_path='',
            static_folder='web/static',
            template_folder='web/templates')


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    os.system("flask --app 7_run_adviser_for_investors --debug run")
