---
title: Cat-Dog-Pandas Classifier 🐾
emoji: 🐶🐱🐼
colorFrom: indigo
colorTo: blue
sdk: docker
python_version: "3.9"
app_file: app.py
app_port: 7860
pinned: false
short_description: "Streamlit app to classify Cat, Dog, or Panda images."
tags:
  - streamlit
  - pytorch
  - docker
  - computer-vision
  - deep-learning
thumbnail: "https://huggingface.co/front/thumbnails/space-docker-streamlit.png"
---


# 🐾 Cat-Dog-Pandas Classifier

A **Streamlit** web app that uses **PyTorch** and **transfer learning (AlexNet)** to classify images of **Cats 🐱**, **Dogs 🐶**, and **Pandas 🐼** in real time.  
Deployed using **Docker** on **Hugging Face Spaces**.

---

## 🧠 Model Overview

- **Base Model:** AlexNet (Transfer Learning)  
- **Fine-tuned Layers:** Layer4 + Fully Connected layers  
- **Custom Classifier:**  
  - Linear(2048 → 512) + ReLU + Dropout(0.7)  
  - Linear(512 → 128) + ReLU + Dropout(0.3)  
  - Linear(128 → 3) [Output Layer]  
- **Classes:** Cat, Dog, Panda  
- **Auto-Download:** Model (`model.pth`) fetched automatically from Hugging Face  
- **Device Support:** CPU/GPU auto-detection  

---

## 📊 Performance Summary

| Metric | Cat | Dog | Panda | Macro Avg | Weighted Avg |
|:--|:--:|:--:|:--:|:--:|:--:|
| **Precision** | 98.51% | 99.49% | 100.00% | 99.33% | 99.33% |
| **Recall** | 99.50% | 98.50% | 100.00% | 99.33% | 99.33% |
| **F1-Score** | 99.00% | 98.99% | 100.00% | 99.33% | 99.33% |
| **Test Accuracy** | – | – | – | **99.33%** | – |

---

## 📁 Project Structure

Cat-Dog-Pandas/  
├── 🐳 Dockerfile  
├── 📱 app.py  
├── 📈 metrics.json  
├── 🧠 model.pth  
├── 📓 Project.ipynb  
├── 📋 requirements.txt  
├── 🖼️ confusion_matrix.png  
└── 🔧 datasplit.py  


---

## ⚙️ Dockerfile (Used in this Project)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0","--server.enableXsrfProtection=false","--server.enableCORS=false","--server.maxUploadSize=50"]
---
```

## Author:
Developed by Mahesh Raj Purohit
