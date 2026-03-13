# 🧠 Visage AI — Deep Face Analytics & Hybrid Image Retrieval

<div align="center">

![Visage AI Banner](https://img.shields.io/badge/Visage-AI-6366f1?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end AI system for face detection, attribute prediction, deep embedding extraction, and hybrid image retrieval — served via a production-grade FastAPI backend and a stunning React frontend.**

[Features](#-features) • [System Pipeline](#-system-pipeline) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Training](#-training-on-kaggle) • [API Docs](#-api-reference) • [Screenshots](#-screenshots)

</div>

---

## ✨ Features

- 🔍 **Real-time Face Detection** — MTCNN-based multi-face detection with landmark extraction
- 👤 **Age & Gender Prediction** — Dual-head ResNet18 fine-tuned on UTKFace (±4.6 yr MAE, 93% gender accuracy)
- 😄 **Emotion Recognition** — 7-class ResNet18 trained on FER2013 (68%+ accuracy)
- 🏷️ **40 Facial Attributes** — CelebA attribute predictor (glasses, beard, hair color, etc.)
- 🔗 **Deep Face Embeddings** — FaceNet InceptionResnetV1 pretrained on VGGFace2
- ⚡ **Hybrid Vector Search** — FAISS cosine similarity + SQLite attribute filtering
- 🔥 **Grad-CAM Heatmaps** — Visual explainability for emotion prediction
- 🚀 **FastAPI Backend** — Async, production-ready REST API with Pydantic validation
- 🎨 **Premium React UI** — Glassmorphism dark theme with Framer Motion animations

---

## 🔄 System Pipeline

```
Upload Image
     │
     ▼
Face Detection (MTCNN)
     │
     ├──► Face Alignment & Crop
     │
     ├──► Attribute Prediction
     │         ├── Age & Gender  (ResNet18 / UTKFace)
     │         ├── Emotion       (ResNet18 / FER2013)
     │         └── 40 Attributes (ResNet18 / CelebA)
     │
     ├──► Deep Embedding Extraction (InceptionResnetV1 / VGGFace2)
     │
     ├──► Hybrid Retrieval
     │         ├── Attribute Filter  (SQLite)
     │         └── Cosine Similarity (FAISS)
     │
     └──► Grad-CAM Explainability Heatmap
```

---

## 🛠 Tech Stack

### Backend
| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Face Detection | `facenet-pytorch` (MTCNN) |
| Embeddings | `facenet-pytorch` (InceptionResnetV1) |
| Attribute Models | PyTorch + ResNet18 |
| Vector Search | FAISS (CPU) |
| Metadata Store | SQLite |
| Explainability | Captum (Grad-CAM) |
| Image Processing | OpenCV, Pillow |
| Config | Pydantic Settings |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 + Vite |
| Animations | Framer Motion |
| HTTP Client | Axios |
| Icons | Lucide React |
| Styling | Vanilla CSS (Design Tokens) |
| Typography | Inter + Outfit (Google Fonts) |

---

## 📁 Project Structure

```
visage-ai/
├── app/                         # FastAPI Backend
│   ├── api/
│   │   ├── routes.py            # API endpoints
│   │   └── schemas.py           # Pydantic models
│   ├── core/
│   │   ├── config.py            # Centralized settings
│   │   └── device.py            # CUDA/CPU auto-detection
│   ├── models/
│   │   ├── age_gender_model.py  # Dual-head ResNet18
│   │   ├── emotion_model.py     # 7-class ResNet18
│   │   ├── attribute_model.py   # 40-attribute ResNet18
│   │   └── embedding_model.py   # FaceNet InceptionResnetV1
│   ├── pipeline/
│   │   └── face_pipeline.py     # End-to-end orchestrator
│   ├── retrieval/
│   │   ├── faiss_index.py       # FAISS vector index wrapper
│   │   ├── sqlite_store.py      # Metadata database
│   │   └── hybrid_search.py     # Combined retrieval engine
│   ├── utils/
│   │   ├── image_utils.py       # Image preprocessing
│   │   └── visualization.py     # Heatmap overlay rendering
│   └── main.py                  # FastAPI app entry point
│
├── training/                    # Model Training Scripts
│   ├── datasets/
│   │   ├── utkface.py           # UTKFace PyTorch Dataset
│   │   ├── fer2013.py           # FER2013 PyTorch Dataset
│   │   └── celeba.py            # CelebA PyTorch Dataset
│   ├── train_age_gender.py      # UTKFace training
│   ├── train_emotion.py         # FER2013 training
│   └── train_attributes.py      # CelebA training
│
├── notebooks/                   # Kaggle-ready Training Notebooks
│   ├── kaggle_train_emotion.py
│   ├── kaggle_train_age_gender.py
│   └── kaggle_train_attributes.py
│
├── frontend/                    # React Frontend (Vite)
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalyzeView.jsx  # Face analysis upload + results
│   │   │   └── SearchView.jsx   # Hybrid search with filters
│   │   ├── App.jsx              # Layout + navigation
│   │   ├── App.css              # Component styles
│   │   ├── index.css            # Design tokens + global styles
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
│
├── data/
│   └── weights/                 # Trained model weights (not tracked)
│       ├── emotion_model.pth
│       ├── age_gender_model.pth
│       └── attribute_model.pth
│
├── tests/
│   └── test_pipeline.py         # End-to-end pipeline tests
│
├── requirements.txt
├── start.bat                    # One-click Windows launcher
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11 (recommended, 3.13 has some wheel issues)
- Node.js 18+
- Windows / Linux / macOS

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/visage-ai.git
cd visage-ai
```

### 2. Install Python dependencies
```bash
pip install torch torchvision facenet-pytorch faiss-cpu fastapi uvicorn \
    python-multipart opencv-python-headless Pillow numpy pandas \
    captum aiofiles pydantic pydantic-settings scipy scikit-learn tqdm
```

### 3. Install Frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Start the application (Windows)
```bash
.\start.bat
```

This launches:
- **Backend API** → http://localhost:8000
- **React Frontend** → http://localhost:5173
- **Interactive API Docs** → http://localhost:8000/docs

---

## 🎓 Training on Kaggle (Free GPU)

The models need to be fine-tuned on labeled face datasets for best accuracy. We provide ready-to-run Kaggle notebooks for all three models.

### Model 1 — Emotion Recognition (FER2013)
1. Open: [kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click **"New Notebook"**
3. Enable GPU: `Settings → Accelerator → GPU T4 x2`
4. Paste contents of `notebooks/kaggle_train_emotion.py` → Run All
5. Download `emotion_model.pth` → place in `data/weights/`

**Results achieved:** Val Accuracy **68.1%** @ Epoch 3/30 (FER2013 SOTA: ~73%)

### Model 2 — Age & Gender (UTKFace)
1. Open: [kaggle.com/datasets/jangedoo/utkface-new](https://www.kaggle.com/datasets/jangedoo/utkface-new)
2. Follow the same notebook process with `notebooks/kaggle_train_age_gender.py`
3. Download `age_gender_model.pth` → place in `data/weights/`

**Results achieved:** Age MAE **4.63 years** | Gender Accuracy **93.36%**

### Model 3 — 40 Facial Attributes (CelebA)
1. Open: [kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
2. Follow the same notebook process with `notebooks/kaggle_train_attributes.py`
3. Download `attribute_model.pth` → place in `data/weights/`

> **Note:** Without trained weights, the system falls back to ImageNet pretrained backbone (reduced accuracy). The FaceNet embedding model is always accurate as it uses official VGGFace2 pretrained weights.

---

## 📡 API Reference

### `POST /api/analyze`
Analyze a face image — detect, align, predict attributes, and extract embeddings.

**Request:** `multipart/form-data`
| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image file (JPEG, PNG, WEBP) |
| `generate_heatmap` | bool | `true` | Generate Grad-CAM overlay |
| `top_k` | int | `5` | Number of similar faces to retrieve |

**Response:**
```json
{
  "detected_image": "<base64 JPEG with bounding boxes>",
  "faces": [
    {
      "face_id": 0,
      "age": 27,
      "gender": "female",
      "gender_confidence": 0.9481,
      "emotion": "happy",
      "emotion_all_scores": { "happy": 0.87, "neutral": 0.08, ... },
      "present_attributes": ["Smiling", "High_Cheekbones", "Wavy_Hair"],
      "heatmap_emotion": "<base64 JPEG heatmap>",
      "embedding": [0.0123, -0.0456, ...]
    }
  ]
}
```

### `POST /api/search`
Search the indexed face database using attribute filters + embedding similarity.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | File (optional) | Query face image for embedding similarity |
| `gender` | str (optional) | `"male"` or `"female"` |
| `emotion` | str (optional) | e.g. `"happy"`, `"sad"` |
| `age_min` | int (optional) | Minimum age filter |
| `age_max` | int (optional) | Maximum age filter |
| `top_k` | int | Number of results to return |

### `POST /api/build-database`
Index a folder of images into FAISS + SQLite for search.

**Request body (JSON):**
```json
{ "image_dir": "C:/path/to/your/face/images" }
```

### `GET /api/health`
Health check endpoint — returns API status and model load state.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 🏗 Architecture Decisions

### Why ResNet18?
- Lightweight enough to run on a consumer GPU (4GB VRAM)
- Sufficient expressiveness for binary/small categorical predictions
- Fast inference (~5-10ms per face)
- Easy to extend heads for multi-task learning

### Why FAISS for vector search?
- Facebook's battle-tested vector index library
- Sub-millisecond approximate nearest-neighbor search on millions of vectors
- CPU-compatible, no GPU required for inference
- Persistent index with `IndexFlatIP` for cosine similarity

### Why Hybrid Retrieval?
Pure embedding search can return irrelevant results if the query embedding is ambiguous. Pre-filtering by hard attributes (gender, age range, emotion) dramatically reduces the candidate pool before cosine similarity ranking.

### Why Grad-CAM?
Model explainability is essential for production AI systems. Grad-CAM generates human-readable attention maps using gradients of the predicted class score with respect to the last convolutional layer activations — no model modification required.

---

## 📊 Model Performance Summary

| Model | Dataset | Metric | Score |
|---|---|---|---|
| Emotion | FER2013 | Val Accuracy | 68.1% |
| Age | UTKFace | Val MAE | 4.63 years |
| Gender | UTKFace | Val Accuracy | 93.4% |
| Attributes | CelebA | Avg Attr Accuracy | ~90%+ |
| Embedding | VGGFace2 | LFW Accuracy | 99.65% (pretrained) |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first for major changes.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN & InceptionResnetV1
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — Emotion dataset
- [UTKFace](https://susanqq.github.io/UTKFace/) — Age & Gender dataset  
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — Facial attributes dataset
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search library
- [Captum](https://captum.ai/) — Model interpretability

---

<div align="center">
Built with ❤️ using PyTorch, FastAPI, and React
<br/>
<strong>Visage AI</strong> — See beyond the surface.
</div>
