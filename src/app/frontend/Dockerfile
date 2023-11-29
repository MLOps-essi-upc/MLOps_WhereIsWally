FROM python:3.9-slim

WORKDIR /web

COPY ./src/app/frontend/web .

RUN pip install --upgrade pip
RUN pip install Django
RUN pip install requests
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
