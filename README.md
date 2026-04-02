# Handwritten Script Identification Across Multiple Alphabets

This project investigates how well neural network architectures generalize across different writing systems for handwritten text recognition. Specifically, we compare performance on **Latin (French)** and **Cyrillic (Kazakh/Russian)** scripts using a unified model architecture.

---

## 🚀 Overview

Most OCR systems are designed for a single script (often Latin). However, many real-world applications require handling **multiple scripts**, especially in multilingual regions.

This project explores:
- Whether a single model can generalize across scripts
- How performance differs between Latin and Cyrillic handwriting
- Common sources of recognition errors

---

## 📊 Datasets

We trained and evaluated on two datasets:

### 1. Kazakh Offline Handwritten Text Dataset (KOHTD)
- ~140K samples (mostly Kazakh, some Russian)
- Word-level annotations
- Preprocessed to **32×128 grayscale images**

### 2. French Handwritten Archive Dataset
- ~12K samples
- Sentence-level annotations
- Variable width images (fixed height = 32px)

Both datasets were split into **80% train / 10% validation / 10% test**.

---

## 🧠 Model Architecture

We implemented a **CRNN (CNN + BiLSTM + CTC)** pipeline:

- **CNN (5 layers)**  
  - Extracts visual features from images  
  - Uses batch normalization, ReLU, and max pooling  

- **Bidirectional LSTM (2 layers)**  
  - Captures sequential dependencies in text  
  - Processes features left-to-right and right-to-left  

- **CTC Loss + Greedy Decoder**  
  - Converts frame-level predictions into text sequences  

### Training Details
- Learning rate: 0.001  
- Batch size: 512  
- Epochs: 50  
- Dropout: 30% (LSTM)

---

## 📈 Results

| Metric                  | French (Latin) | Kazakh (Cyrillic) |
|------------------------|----------------|-------------------|
| Character Accuracy     | 0.8983         | 0.8902            |
| Character Error Rate   | 0.1017         | 0.1080            |
| Word Accuracy          | 0.6121         | 0.6045            |

### Key Findings
- Performance is **very similar across scripts**
- Model generalizes well between Latin and Cyrillic
- No significant script-specific degradation

---

## 🔍 Analysis

### Misclassifications
- Errors mainly occur between **visually similar characters**
- Common confusion between uppercase/lowercase and similar shapes
- Issues are **not script-specific**, but universal to handwriting

### Effect of Word Length
- Accuracy **decreases as word/sentence length increases**
- Longer sequences accumulate more prediction errors

### Model Attention (Grad-CAM)
- Model focuses correctly on **stroke regions and character shapes**
- Similar attention patterns across both scripts
- Misclassifications are **not due to lack of attention**

---

## ⚠️ Limitations

- Limited hyperparameter tuning due to compute constraints
- Smaller dataset for French compared to Kazakh
- Performance drops on **longer words and rare characters**
- Some overfitting observed (train vs test gap)

---

## 🔮 Future Work

- Experiment with **Transformer-based OCR models** (e.g., TrOCR)
- Apply **data augmentation** to improve robustness
- Improve handling of **longer sequences**
- Use **transfer learning / multilingual pretraining**
- Expand to additional scripts and low-resource languages

---

## 🤝 Contributions

- **Amira Mahmedjan**: Model training, CAM visualizations, literature review  
- **Stephanie Hurtado**: Evaluation metrics, performance analysis, cross-dataset comparison  

---

## 📚 References

Key works include research on CRNN models, transformer-based OCR (TrOCR), and handwritten text datasets such as KOHTD.

---

## 📌 Takeaway

A standard CRNN-based OCR model can **generalize effectively across fundamentally different writing systems**, suggesting strong potential for building **multi-script, inclusive text recognition systems**.

## Project Poster: 
https://docs.google.com/presentation/d/1dkppOwtBHFSBO8ZKAJaZwOo_PDN8sf0rZgFJQhq18GQ/edit?usp=sharing
