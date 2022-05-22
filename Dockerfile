FROM continuumio/miniconda3:latest

RUN pip install mlflow boto3 pymysql

ADD . /app
WORKDIR /app

COPY time_sc.sh time_sc.sh 
RUN chmod +x time_sc.sh