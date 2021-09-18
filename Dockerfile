FROM python:3.7.11-slim-buster

WORKDIR /usr/src/app

COPY . .
RUN pip install --no-cache-dir --user -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:."
ENV PYTHONUNBUFFERED 1