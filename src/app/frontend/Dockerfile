FROM python:3.9-slim

WORKDIR /web

COPY ./web .

RUN pip install --upgrade pip
RUN pip install Django
RUN pip install requests
RUN pip install opencv-python

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
