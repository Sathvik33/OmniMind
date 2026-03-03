FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/user.py", "--server.port=8501", "--server.address=0.0.0.0"]