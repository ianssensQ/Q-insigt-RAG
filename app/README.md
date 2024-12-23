В проекте используется облачная база данных PostgreSQL (Работает постоянно). 
Однако для старта также требуется инициализация бд и парсера.
Они находятся в   

- __*./app/start/db_init*__ - (требуется для запуска только в первый раз)
- __*./app/services/rabbit/utils/parser_init*__ - (требуется для обновления сессии подключения)


Может потребоваться запустить парсинг сесии (рега при запуске с нуля)в других местах ( почему-то не всегда ???):
- __*./app/tg_bot/handlers/*__
- __*./app/tg_bot/*__
- __*./app/services/rabbit/workers/*__ 

После первого входа информация о сессии сохраняется.

Для классификации
python -m spacy download ru_core_news_sm







---
23.12.24 (last_version)

**Порядок действий**

0. Заполните `.env` файл по аналогии с `.env.template`.  
1. В файле `app\services\rabbit\utils\wait_for_rabbitmq.sh` выберите в правом нижнем углу LF, если не стоит.
2. Запустите ` docker-compose up -d --no-deps --build parser_init `. Получите код подтверждения в Telegram.
3. Остановите контейнер (Ctrl+C).
4. Добавьте код в `docker-compose.yml` в `parser_init`:
CONFIRMATION_CODE: "12345"
5. Перезапустите контейнер:
`docker-compose up --build`
