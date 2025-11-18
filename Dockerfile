# Dockerfile (root)
FROM python:3.10-slim

WORKDIR /app

# system deps for pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app and model (ensure model/savedmodel.pth exists before building or you can copy in CI)
COPY . .

EXPOSE 5000
ENV FLASK_APP=app/app.py
CMD ["python", "app/app.py"]