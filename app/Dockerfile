FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    net-tools \
    iputils-ping \
    curl \
    telnet \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords

RUN python -m spacy download ru_core_news_sm

COPY services/rabbit/utils/wait_for_rabbitmq.sh /wait_for_rabbitmq.sh
RUN chmod +x /wait_for_rabbitmq.sh

COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["/wait_for_rabbitmq.sh", "rabbitmq", "5672", "python", "-m", "services.rabbit.workers.all_summ_workers"]
