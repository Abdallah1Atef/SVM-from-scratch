# SVM from Scratch: Banknote Authentication
This project implements a **Support Vector Machine (SVM) classifier from scratch** to authenticate banknotes using wavelet transform features. Unlike library implementations, this version:

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Machine%20Learning-SVM-orange)

A from-scratch implementation of Support Vector Machines (SVM) using both gradient descent (logistic loss) and subgradient descent (hinge loss) methods for classifying authentic and forged banknotes.

## Table of Contents
- [Features](#features)
- [Implementation](#implementation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Features
- 🛠️ **Custom SVM Implementation**:
  - Gradient descent with logistic loss
  - Subgradient descent with hinge loss
- 📊 **Comprehensive Evaluation**:
  - Accuracy, precision, recall metrics
  - Training/validation loss curves
  - Confusion matrices
- ⚡ **Optimization**:
  - Early stopping based on convergence
  - L2 regularization
- 🔍 **Data Processing**:
  - Manual standardization
  - Custom train-val-test split

## 🧠 Implementation

### 🔑 Key Components

#### Loss Functions
- **Hinge Loss** (for subgradient descent):
  ```python
  def hinge_loss(w, b, X, y, reg=0.01):
      margins = 1 - y * (np.dot(X, w) + b)
      return np.mean(np.maximum(0, margins)) + reg * np.dot(w, w)
  ```
- **Logistic Loss** (for gradient descent):
  ```python
  def logistic_loss(w, b, X, y, reg=0.01):
    scores = y * (np.dot(X, w) + b)
    return np.mean(np.log(1 + np.exp(-scores))) + reg * np.dot(w, w)
  ```
## Optimization 

-Learning Rate: Fixed rate (0.01) with early stopping
-Regularization: L2 penalty (λ = 0.01)
-Convergence: Tolerance threshold of 1e-4
-Evaluation Metrics
-Accuracy: (TP + TN) / (TP + TN + FP + FN)
-Precision: TP / (TP + FP)
-Recall: TP / (TP + FN)
-Confusion Matrix visualization

-Results
-Metric	Gradient Descent	Subgradient Descent
-Test Accuracy	 98.18%	        99.27%
-Training Loss	 0.143	        0.148
-Precision	     0.98 	        0.99
-Recall	         0.97          	0.98

-Key Findings
🚀 Subgradient method achieved better final accuracy (99.27% vs 98.18%)

📈 Both methods show excellent generalization with <1% performance drop

🔄 Gradient descent has smoother convergence due to differentiable objective

⏱️ Subgradient descent shows faster initial convergence
### Core Algorithms
```python
def train_svm(X, y, X_val, y_val, method="gradient", lr=0.01, epochs=1000, reg=0.01):
    """Train SVM with either gradient or subgradient descent"""
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0
    loss_hist, acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # Compute gradient and losses
        if method == "gradient":
            dw, db = logistic_gradient(w, b, X, y, reg)
            train_loss = logistic_loss(w, b, X, y, reg)
            val_loss = logistic_loss(w, b, X_val, y_val, reg)
        elif method == "subgradient":
            margins = 1 - y * (X.dot(w) + b)
            mask = margins > 0
            #(max for the requlerization equation )
            grad_contrib = (-y[:, None] * X) * mask[:, None]
            #(result from differatiation of the  regulaization equation acording to W (weights))
            dw = np.mean(grad_contrib, axis=0) + 2 * reg * w
            #(result from differatiation of the  regulaization equation acording to b )
            db = np.mean(-y * mask)
            train_loss = hinge_loss(w, b, X, y, reg)
            val_loss = hinge_loss(w, b, X_val, y_val, reg)
        else:
            raise ValueError("Unknown method")

        # Update parameters
        w -= lr * dw
        b -= lr * db

        # Record metrics
        loss_hist.append(train_loss)
        acc = np.mean(np.sign(X.dot(w) + b) == y)
        acc_hist.append(acc)
        val_loss_hist.append(val_loss)
        val_acc = np.mean(np.sign(X_val.dot(w) + b) == y_val)
        val_acc_hist.append(val_acc)

        # Early stopping: check loss convergence
        if epoch > 0 and abs(val_loss_hist[-1] - val_loss_hist[-2]) <= tol:
            break

    return w, b, loss_hist, acc_hist, val_loss_hist, val_acc_hist
```
## Visualization
![image](https://github.com/user-attachments/assets/bf9fc4e7-92c8-4418-99b2-b973cb21923b)
![image](https://github.com/user-attachments/assets/e15d41e3-49c1-49ee-9865-d852ee330b5c)
![image](https://github.com/user-attachments/assets/af101e97-4585-4a11-b958-07ed1301d83d)



🔧 **Core Features**:
- Dual optimization approaches:  
  - *Gradient Descent* with logistic loss (smooth optimization)  
  - *Subgradient Descent* with hinge loss (efficient for non-differentiable functions)  
- Full machine learning pipeline:  
  - Custom data standardization  
  - Manual train-val-test splitting  
  - L2 regularization  
- Comprehensive evaluation:  
  - Loss/accuracy tracking  
  - Confusion matrices  
  - Precision-recall metrics  

🎯 **Purpose**:  
Demonstrates fundamental ML concepts by:  
1. Building SVM math from first principles  
2. Comparing optimization methods' performance  
3. Highlighting model interpretability  

📊 **Dataset**:  
Uses the [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) containing:  
- 1,372 instances  
- 4 wavelet features (variance, skewness, kurtosis, entropy)  
- Binary classification (authentic/forged)  

🛠️ **Tech Stack**:  
```python
Python 3.8+ | NumPy | Matplotlib | scikit-learn (metrics only) | pandas
