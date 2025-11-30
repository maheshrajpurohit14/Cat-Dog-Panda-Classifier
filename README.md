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
├── 📓 Project.ipynb  
├── 📋 requirements.txt  
├── 🖼️ confusion_matrix.png  
└── 🔧 datasplit.py  


---

## ⚙️ Project.py (Used in this Project)

```
num_epochs = 20
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_acc = 0.0
start_time  = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    # Wrap train_loader with tqdm
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
    
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update tqdm bar postfix with running metrics
        train_loader_tqdm.set_postfix({'loss': f"{loss.item():.4f}"})

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
    
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            val_loader_tqdm.set_postfix({'loss': f"{loss.item():.4f}"})

    val_loss /= total_val
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    
    # Final log for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model.pth')
        print(f"💾 Saved best model with val_acc: {val_acc:.4f}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"⏹ Early stopping triggered at epoch {epoch+1}")
        break

end_time = time.time()
print(f"\nTraining completed in {(end_time - start_time)/60:.1f} minutes")
---
```

## Author:
Developed by Mahesh Raj Purohit
