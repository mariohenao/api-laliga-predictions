FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install tensorflow==2.7

RUN pip install pandas==1.1.5

RUN pip install numpy==1.19.5

COPY ./model /model/

COPY ./app /app/

COPY ./data /data/

EXPOSE 8080

CMD ["python", "main.py"]