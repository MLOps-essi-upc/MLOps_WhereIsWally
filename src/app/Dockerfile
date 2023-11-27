FROM python:3.9-slim
WORKDIR /api
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt
EXPOSE 5000
ENTRYPOINT [ "uvicorn", "app:app", "--host", "0.0.0.0", "--reload" ]
