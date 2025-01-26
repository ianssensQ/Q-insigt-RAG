FROM python:3.11-slim

COPY requirements.txt /workdir/
COPY app/ /workdir/app/
COPY model/ /workdir/model/

WORKDIR /workdir

VOLUME ["/workdir/logs"]

RUN mkdir -p /workdir/logs && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader -d /usr/share/nltk_data stopwords

ENV NLTK_DATA=/usr/share/nltk_data

EXPOSE 8080
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]