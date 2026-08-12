FROM python:3.11-slim

WORKDIR /app

# don't buffer stdout, and don't write .pyc into the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# dependencies first: this layer is only rebuilt when requirements.txt changes,
# so editing application code doesn't trigger a full reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# application code, the trained pipeline, and the dashboard.
# .dockerignore keeps the raw CSV and train.py out of model/.
COPY main.py .
COPY model/ ./model/
COPY frontend/ ./frontend/

# fail the build now, loudly, rather than at container start if the pickle is missing
RUN test -f model/churn_model.pkl || (echo "ERROR: model/churn_model.pkl not found — run model/train.py before building" && exit 1)

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
