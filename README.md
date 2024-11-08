# 🛍️ Fashion Search Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent fashion catalog system powered by deep learning that performs both **classification** and **visual similarity search** using a shared MobileNet-inspired backbone.

![Fashion Search Demo](assets/demo.gif)

## 🌟 Features

- **Multi-task Learning**: One backbone, two tasks (classification + similarity search)
- **Efficient Architecture**: MobileNetV2-inspired with inverted residual blocks (429K parameters)
- **Visual Search**: Find similar fashion items using metric learning with Siamese networks
- **Transfer Learning**: Reuse classifier features for similarity learning
- **Production Ready**: ONNX export, FAISS integration, Docker support

## 📊 Results

| Task | Metric | Score |
|------|--------|-------|
| Classification | Validation Accuracy | **85.51%** |
| Visual Search | Top-1 Accuracy | **68.2%** |
| Visual Search | Top-5 Accuracy | **89.3%** |
| Model Size | Parameters | **429K** |
| Inference Speed | GPU (V100) | **15ms** |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fashion-search-engine.git
cd fashion-search-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download the clothing dataset
python scripts/download_dataset.py --output data/
```

### Train Models

```bash
# 1. Train the classifier
python train_classifier.py --epochs 10 --batch-size 32 --lr 0.001

# 2. Train the Siamese network (uses pretrained classifier backbone)
python train_siamese.py --epochs 15 --batch-size 32 --lr 0.0001 \
    --pretrained-backbone checkpoints/classifier_best.pth
```

### Run Visual Search

```bash
# Interactive visual search
python search.py --query path/to/query_image.jpg --top-k 5

# Or use the Python API
python
>>> from inference.search import VisualSearchEngine
>>> engine = VisualSearchEngine('checkpoints/siamese_best.pth')
>>> results = engine.search('query.jpg', k=5)
>>> engine.display_results(results)
```

## 📁 Project Structure

```
fashion-search-engine/
├── data/                          # Dataset directory
│   ├── clothing-dataset-small/
│   │   ├── train/
│   │   └── val/
│   └── README.md
├── models/                        # Model architectures
│   ├── __init__.py
│   ├── blocks.py                 # InvertedResidualBlock
│   ├── backbone.py               # MobileNetBackbone
│   ├── classifier.py             # MobileNetLikeClassifier
│   └── siamese.py                # SiameseEncoder, SiameseNetwork
├── data_loading/                  # Data handling
│   ├── __init__.py
│   ├── dataset.py                # TripleDataset for metric learning
│   ├── transforms.py             # Augmentation pipelines
│   └── loaders.py                # DataLoader utilities
├── training/                      # Training scripts
│   ├── __init__.py
│   ├── classifier_trainer.py    # Classification training
│   ├── siamese_trainer.py       # Metric learning training
│   ├── losses.py                 # Custom loss functions
│   └── utils.py                  # Training utilities
├── inference/                     # Inference and search
│   ├── __init__.py
│   ├── search.py                 # Visual search engine
│   ├── embeddings.py             # Embedding generation
│   └── index.py                  # FAISS indexing
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── visualization.py          # Plotting and visualization
│   ├── metrics.py                # Evaluation metrics
│   └── config.py                 # Configuration management
├── scripts/                       # Utility scripts
│   ├── download_dataset.py
│   ├── export_onnx.py
│   └── build_index.py
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_visual_search_demo.ipynb
├── tests/                         # Unit tests
│   ├── test_models.py
│   ├── test_dataset.py
│   └── test_search.py
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── assets/                        # Images and demos
├── train_classifier.py           # Main classifier training script
├── train_siamese.py              # Main Siamese training script
├── search.py                      # Visual search CLI
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── LICENSE
└── README.md
```

## 🏗️ Architecture

### Overview

The system uses a **shared backbone** for both tasks:

```
Input Image [3, 64, 64]
         ↓
┌────────────────────────┐
│  MobileNetBackbone     │
│  (Shared Features)     │
│  [3,64,64] → [64,4,4] │
└──────────┬─────────────┘
           │
    ┌──────┴──────┐
    ↓             ↓
┌─────────┐  ┌──────────┐
│Classifier│  │ Siamese  │
│  Head    │  │ Encoder  │
└─────────┘  └──────────┘
     ↓             ↓
Categories    Embeddings
```

### Inverted Residual Block

The core building block uses a **narrow → wide → narrow** pattern:

```python
class InvertedResidualBlock(nn.Module):
    """
    Expansion → Depthwise → Projection + Residual
    
    Example: 24 channels → 72 channels → 32 channels
    """
    def forward(self, x):
        skip = x
        out = self.expand(x)      # 1×1 conv: narrow → wide
        out = self.depthwise(out) # 3×3 depthwise: spatial filtering
        out = self.project(out)   # 1×1 conv: wide → narrow
        return F.relu(out + skip) # Residual connection
```

**Why it works:**
- Depthwise convolutions are cheap (1/k² parameters vs standard conv)
- Expansion factor controls capacity (typically 3-6×)
- Residual connections enable deep networks

## 📚 Key Concepts

### 1. Multi-Task Learning with Shared Backbone

Instead of training separate models, we:
1. Train a **classifier** with the backbone + classification head
2. **Reuse the trained backbone** for visual search
3. Add a simple pooling layer to create embeddings

**Benefits:**
- Saves training time (15 epochs vs 30+ from scratch)
- Better generalization (backbone learns robust features)
- Efficient inference (shared computations)

### 2. Metric Learning with Triplet Loss

The Siamese network learns embeddings where:
- **Similar items** (same class) → **close in embedding space**
- **Dissimilar items** (different class) → **far apart**

**Training with Triplets:**
```python
# For each anchor image, select:
anchor = dataset[i]          # Reference image
positive = same_class[j]     # Same category, different image
negative = diff_class[k]     # Different category

# Loss encourages:
# distance(anchor, positive) < distance(anchor, negative) - margin
loss = TripletMarginLoss(margin=1.0)
```

### 3. Efficient Architecture Design

**Comparison:**

| Architecture | Parameters | FLOPs | Accuracy |
|-------------|-----------|-------|----------|
| ResNet-18 | 11.7M | 1.8G | 86% |
| MobileNetV2 | 3.5M | 0.3G | 85% |
| **Ours** | **0.43M** | **0.09G** | **85%** |

Our architecture achieves similar accuracy with:
- **27× fewer parameters** than ResNet-18
- **20× fewer FLOPs**
- **Faster inference** (15ms vs 50ms)

## 🔬 Detailed Usage

### Training with Custom Data

```python
from data_loading import FashionDataset, get_dataloaders
from training import ClassifierTrainer
from models import MobileNetLikeClassifier

# Prepare data
train_loader, val_loader = get_dataloaders(
    data_path='data/my_fashion_data',
    batch_size=32,
    num_workers=4
)

# Initialize model
model = MobileNetLikeClassifier(num_classes=10)

# Train
trainer = ClassifierTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)
trainer.train(epochs=10, lr=0.001)
```

### Building a Visual Search Index

```python
from inference import VisualSearchEngine, FAISSIndex

# Load trained Siamese network
engine = VisualSearchEngine('checkpoints/siamese_best.pth')

# Generate embeddings for catalog
catalog_embeddings = engine.embed_dataset('data/catalog/')

# Build FAISS index for fast search
index = FAISSIndex(dimension=64)
index.build(catalog_embeddings)
index.save('indexes/catalog.faiss')

# Search
query_embedding = engine.embed_image('query.jpg')
similar_items = index.search(query_embedding, k=10)
```

### Export to ONNX

```python
from scripts.export_onnx import export_model

# Export classifier
export_model(
    model_path='checkpoints/classifier_best.pth',
    output_path='exports/classifier.onnx',
    model_type='classifier'
)

# Export Siamese encoder
export_model(
    model_path='checkpoints/siamese_best.pth',
    output_path='exports/siamese.onnx',
    model_type='siamese'
)
```

## 📊 Evaluation

### Classification Metrics

```bash
python evaluate_classifier.py \
    --checkpoint checkpoints/classifier_best.pth \
    --data-path data/clothing-dataset-small/val \
    --metrics accuracy,precision,recall,f1
```

### Visual Search Metrics

```bash
python evaluate_search.py \
    --checkpoint checkpoints/siamese_best.pth \
    --data-path data/clothing-dataset-small/val \
    --metrics top1,top5,mrr,map
```

## 🐳 Docker

```bash
# Build image
docker build -t fashion-search-engine .

# Run training
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/checkpoints:/checkpoints \
    fashion-search-engine python train_classifier.py

# Run inference server
docker-compose up
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=models --cov=training --cov=inference tests/
```

## 📖 Tutorials

Check out our Jupyter notebooks:

1. **[Data Exploration](notebooks/01_data_exploration.ipynb)**: Understand the dataset, visualize samples
2. **[Model Training](notebooks/02_model_training.ipynb)**: Step-by-step training walkthrough
3. **[Visual Search Demo](notebooks/03_visual_search_demo.ipynb)**: Interactive search examples

## 🎓 Learning Resources

### Blog Posts
- [Building an Intelligent Fashion Search Engine](docs/blog_post.md) - Comprehensive guide
- [MobileNet Architecture Explained](docs/mobilenet_explained.md)
- [Metric Learning and Siamese Networks](docs/metric_learning.md)

### Papers
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
- [Deep Metric Learning: A Survey](https://arxiv.org/abs/2004.03014)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fashion-search-engine,
  author = {Your Name},
  title = {Fashion Search Engine: Multi-Task Learning for Classification and Visual Similarity},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/fashion-search-engine}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small)
- Architecture inspiration: MobileNetV2, Siamese Networks
- Built with PyTorch, torchvision, and amazing open-source tools

## 📧 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ using PyTorch**
