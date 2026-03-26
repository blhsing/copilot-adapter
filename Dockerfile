FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 18080

ENTRYPOINT ["python", "copilot_adapter.py"]
CMD ["serve", "--host", "0.0.0.0"]
