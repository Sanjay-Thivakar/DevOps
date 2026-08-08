FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_NAME=BreastCancerClassifier
ENV MODEL_ALIAS=champion
ENV SCALER_PATH=models/scaler.joblib

# Set work directory
WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files and configuration
COPY src/ /app/src/
COPY params.yaml /app/

# Copy model artifacts, database, and scaler
COPY models/scaler.joblib /app/models/scaler.joblib
COPY mlflow.db /app/
COPY mlruns/ /app/mlruns/

# Expose port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
