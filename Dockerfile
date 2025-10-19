FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*
  
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY etl_from_excel.py app.py DataClean.py /app/
COPY Patient_data /app/Patient_data
COPY data /app/data

EXPOSE 8050

CMD ["python", "app.py"]
