FROM python:3.7-slim-buster

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["app.py"]