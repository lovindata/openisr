FROM python:3.7.11-slim-buster

WORKDIR /usr/src/app

COPY ./requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:."