# Docker Setup
## 1. Build the Docker Image

```bash
docker build -t sanjaythivakar/ml_101_baseline:latest .
```

---

## 2. Run the Docker Container

```bash
docker run -p 8000:8000 sanjaythivakar/ml_101_baseline:latest
```

The API will now be available at

```
http://localhost:8000
```

---

## 3. Push the Image to Docker Hub

Login to Docker Hub

```bash
docker login
```

Push the image

```bash
docker push sanjaythivakar/ml_101_baseline:latest
```

---

## 4. Pull the Image

Anyone can pull the image using

```bash
docker pull sanjaythivakar/ml_101_baseline:latest
```

---

## 5. Run the Pulled Image

```bash
docker run -p 8000:8000 sanjaythivakar/ml_101_baseline:latest
```

---

# API Endpoints

## Home

**GET**

```
/
```

---

## Health Check

**GET**

```
/health
```

Response

```json
{
    "status": "ok"
}
```

---

## Prediction

**POST**

```
/predict
```

Request

```json
{
    "features": [
        5.1,
        3.5,
        1.4,
        0.2
    ]
}
```

Example Response

```json
{
    "predicted_class": "setosa",
    "confidence": 0.99,
    "probabilities": {
        "setosa": 0.99,
        "versicolor": 0.01,
        "virginica": 0.00
    }
}
```

---

# Docker Hub Repository

```
https://hub.docker.com/r/sanjaythivakar/ml_101_baseline
```

---
