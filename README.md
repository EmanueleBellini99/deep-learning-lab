# Deep Learning Lab Assignments
## USI - Università della Svizzera italiana (2024/2025)

This repository contains my assignments for the Deep Learning Lab course at USI (Università della Svizzera italiana). The course focuses on practical implementations of deep learning concepts using PyTorch.

### Course Information
- **University**: USI - Università della Svizzera italiana
- **Course**: Deep Learning Lab
- **Academic Year**: 2024/2025
- **Student**: Emanuele Bellini

### Assignments Overview

#### Assignment 1: Polynomial Regression
- Implementation of polynomial regression using PyTorch
- Key features:
  - Custom dataset creation with controlled noise
  - Data visualization and analysis
  - Implementation of a neural network for polynomial fitting
  - Training process visualization
  - Parameter convergence analysis

#### Assignment 2: Image Classification with CNNs
- Implementation of Convolutional Neural Networks for CIFAR-10 classification
- Key features:
  - Dataset analysis and preprocessing
  - CNN architecture design and implementation
  - Training and validation pipeline
  - Performance visualization
  - Architecture improvements and experimentation
  - Seed sensitivity analysis
 
#### Assignment 3: Language Model with LSTM
- Implementation of a language model using LSTM for headline generation
- Key features:
  - Dataset processing for political headlines
  - LSTM architecture implementation
  - Training with standard and TBPTT approaches
  - Text generation with different sampling strategies
  - Word embedding analysis
  - Performance comparison of training methods

#### Assignment 4: TSP with Transformers
- Implementation of a transformer model for solving the Traveling Salesman Problem
- Key features:
  - Custom dataset handling for TSP instances
  - Transformer architecture implementation
  - Training with gradient accumulation
  - Comparison with baseline algorithms
  - Performance analysis and visualization
  - Solution quality evaluation

### Repository Structure
```
.
├── 1_polynomial_regression/
│   ├── polynomial_regression.py
│   └── polynomial_regression.pdf
├── 2_CNN/
│   ├── CNN.py
│   └── CNN.pdf
├── 3_LSTM/
│   ├── LSTM.py
│   └── LSTM.pdf
├── 4_transformer_tsp/
│   ├── transformers.py
│   └── transformers.pdf
└── README.md
```

### Technical Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- torchvision (for Assignment 2)
- datasets (Hugging Face)
- networkx (for TSP)

### Usage
Each assignment can be run independently. Make sure to have all required dependencies installed.

### Learning Outcomes
- Deep learning model implementations from scratch
- Data preprocessing and visualization techniques
- Advanced model architectures (CNN, LSTM, Transformer)
- Multiple training approaches and optimizations
- Performance analysis and evaluation
- Practical experience with PyTorch ecosystem
- Solutions for both supervised and self-supervised tasks

---
*Note: This repository represents coursework for the Deep Learning Lab course at USI. Future assignments will be added as they are completed.*
