# 🩺 MedVQA: Advanced Medical Visual Question Answering with BLIP & CNN-LSTM

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/julliebwz/ft-blip-MEDVQA)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the official implementation of **"A Comparative Analysis of CNN-LSTM Feature Fusion and Transformer-Based BLIP Architectures for Medical VQA"**. Our project explores cutting-edge architectures for answering clinical questions based on medical imagery, comparing a custom **CNN+LSTM** baseline against a fine-tuned **BLIP (Bootstrapping Language-Image Pre-training)** transformer model.

---

## 🚀 Overview

Medical Visual Question Answering (MedVQA) sits at the intersection of computer vision and natural language processing. This project aims to assist clinicians by automating the interpretation of medical images (X-rays, CTs, MRIs) through natural language questions.

We implemented two distinct architectures:
1.  **Transformer-Based (State-of-the-Art):** A fine-tuned **BLIP** model that leverages pre-trained vision-language understanding.
2.  **CNN+LSTM Feature Fusion (Baseline):** A traditional deep learning pipeline using VGG16/ResNet for visual features and LSTM for textual queries.

[cite_start]Our best-performing model, **BLIP-SLAKE**, achieves superior accuracy in identifying modalities, localizing organs, and detecting abnormalities.

---

## 🏗️ Methodologies

### 1. Transformer-Based Approach (BLIP)
We fine-tuned the `Salesforce/blip-vqa-base` model on specialized medical datasets. This architecture uses a Vision Transformer (ViT) as an image encoder and a BERT-like transformer for text, fusing them to generate answers directly.

* **Base Model:** [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)
* **Fine-tuned Model:** [julliebwz/ft-blip-MEDVQA](https://huggingface.co/julliebwz/ft-blip-MEDVQA)

### 2. CNN-LSTM Feature Fusion
As a robust baseline, we constructed a custom model:
* **Visual Encoder:** VGG16 or ResNet (pre-trained on ImageNet) to extract high-level image features.
* **Language Encoder:** LSTM (Long Short-Term Memory) network to process the question embeddings.
* **Fusion Strategy:** Pointwise multiplication of visual and textual feature vectors, followed by fully connected dense layers for classification.

---

## 📊 Datasets

We utilized two prominent public datasets for training and evaluation:

| Dataset | Description | Hugging Face ID |
| :--- | :--- | :--- |
| **VQA-RAD** | Clinician-generated dataset containing radiology images and QA pairs. | `flaviagiammarino/vqa-rad` |
| **SLAKE** | A comprehensive dataset with rich semantic labels for medical VQA. (English Only) | `mdwiratathya/SLAKE-vqa-english` |

---

## 🛠️ Installation & Setup

To reproduce our results, ensure you have a Python environment ready (Python 3.8+ recommended).


### 1. Clone the Repository
```bash
git clone [https://github.com/julliebwz/MED-VQA-Project.git](https://github.com/julliebwz/MED-VQA-Project.git)
cd MED-VQA-Project

🚀 Quick Start: InferenceYou can use our best-performing model directly from Hugging Face without retraining.Pythonfrom transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# 1. Load Model & Processor
repo_id = "julliebwz/ft-blip-MEDVQA"
processor = BlipProcessor.from_pretrained(repo_id)
model = BlipForQuestionAnswering.from_pretrained(repo_id)

# 2. Load Your Image
image_path = "path/to/xray_image.jpg" 
raw_image = Image.open(image_path).convert("RGB")

# 3. Ask a Question
question = "What abnormality is seen in this image?"

# 4. Generate Answer
inputs = processor(raw_image, question, return_tensors="pt")
output = model.generate(**inputs)
answer = processor.decode(output[0], skip_special_tokens=True)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## 🧪 Reproduction Guide (Training)

We have provided three Jupyter Notebooks corresponding to our experiments. You can run these in Google Colab or a local Jupyter environment.

| Notebook File | Experiment Description | Dataset |
| :--- | :--- | :--- |
| **`BLIP_SLAKE_.ipynb`** | **[Best Model]** Fine-tuning BLIP on the SLAKE dataset. | SLAKE |
| **`BLIP_VQA_RAD.ipynb`** | Fine-tuning BLIP on the VQA-RAD dataset. | VQA-RAD |
| **`CNN+LSTM_VQA_RAD.ipynb`** | Training the baseline CNN-LSTM model. | VQA-RAD |

### Steps to Train:
1.  **Open** the desired `.ipynb` file (e.g., in Google Colab).
2.  **Install Dependencies:** Ensure the first cell (installing libraries) is executed.
3.  **Run All Cells:** The notebooks are configured to automatically download the dataset via the Hugging Face `datasets` library and start the training process.

---

## 👥 Authors & Credits

**Group FX — Advance Machine Learning (WOA7015)**
* **Najla Geis Junaid Bawazier (24068527)**
* **Salsabila Harlen (24076059)**

*University of Malaya*

---

## 📝 Citation

If you use this code or model in your research, please cite the original BLIP paper and the datasets
