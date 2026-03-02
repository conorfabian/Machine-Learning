# ISLP — Chapter 2: Statistical Learning (Structured Notes)

---

# 2.1 What Is Statistical Learning?

Statistical learning is the study of methods for understanding and modeling the relationship between variables using data. In this chapter, the goal is to formalize what we mean by "learning from data" and to introduce the conceptual tools that will be used throughout the rest of the book.

---

## 2.1.1 The Statistical Learning Framework

We observe data consisting of:

- Inputs (features):

```text
X = (X₁, X₂, ..., X_p)
```

- Output (response):

```text
Y
```

Each observation provides a pair (xᵢ, yᵢ), and we collect many such observations to form a training dataset.

We assume a true relationship between inputs and output:

```text
Y = f(X) + ε
```

Where:

- f = unknown systematic component (the true underlying function)
- ε = random noise:

```text
E[ε] = 0,   Var(ε) = σ²
```

The function f represents the deterministic part of the relationship between X and Y, while ε captures randomness, measurement error, omitted variables, or inherent unpredictability.

Goal: Use training data to estimate f and construct an estimator:

```text
f̂
```

The estimated function f̂ is used either to make predictions or to draw conclusions about relationships between variables.

This framework underlies all supervised learning methods in the book.

---

## 2.1.2 Why Estimate f?

There are two primary motivations for estimating f.

### A) Prediction

Goal: Accurately predict Y for new, unseen X.

- Focus: performance on new (test) data
- Interpretability is often secondary
- Flexible models are commonly preferred

In prediction settings, we care about minimizing test error, not necessarily understanding how the model works internally.

Examples:

- Predicting house prices
- Credit risk modeling
- Medical diagnosis
- Image classification

### B) Inference

Goal: Understand the relationship between X and Y.

Typical questions:

- Which predictors are important?
- How large is each effect?
- Is the relationship positive or negative?
- Is the effect statistically significant?

In inference settings, interpretability is crucial. We often prefer simpler, structured models.

Examples:

- Does education increase income?
- Which marketing channel drives sales?
- Does a treatment improve recovery rates?

⚠ Key distinction: Prediction ≠ Inference

A model that predicts extremely well may not be interpretable, and a model that is highly interpretable may not provide the best predictive accuracy.

---

## 2.1.3 How Do We Estimate f?

Given training data:

```text
(x₁, y₁), ..., (x_n, y_n)
```

we choose a statistical learning method that produces:

```text
f̂
```

Different methods vary according to:

- The assumptions they make about f
- Their level of flexibility
- The amount of training data required
- Interpretability
- Computational complexity

### Parametric Methods

Parametric methods assume a specific functional form for f and reduce the learning problem to estimating a finite set of parameters.

Example (Linear Regression):

```text
f(X) = β₀ + β₁X₁ + ... + β_pX_p
```

Steps:

1. Specify the model form
2. Estimate parameters using data

Advantages:

- Conceptually simple
- Requires less data
- Computationally efficient
- Easy to interpret coefficients

Disadvantages:

- Risk of model misspecification
- May have high bias if the true relationship is complex

### Non‑Parametric Methods

Non‑parametric methods do not assume a fixed functional form.

Examples:

- K‑Nearest Neighbors (KNN)
- Splines
- Decision Trees

Advantages:

- Highly flexible
- Can approximate complex nonlinear functions

Disadvantages:

- Require more data to perform well
- More prone to overfitting
- Often harder to interpret

Flexibility is a central theme: more flexible methods can reduce bias but often increase variance.

---

## 2.1.4 Supervised vs Unsupervised Learning

### Supervised Learning

We observe both X and Y.

Tasks include:

- Regression → Y is quantitative
- Classification → Y is categorical

Goal: Predict Y from X.

### Unsupervised Learning

We observe only X.

Goal: Discover structure in the data without labeled outcomes.

Examples:

- Clustering
- Principal Component Analysis (PCA)
- Dimensionality reduction

Unsupervised learning is exploratory in nature.

---

# 2.2 Assessing Model Accuracy

After fitting a model, we must evaluate how well it performs. The key distinction is between training performance and test (generalization) performance.

---

## 2.2.1 Regression Setting

### Training Mean Squared Error (MSE)

```text
Training MSE = (1/n) Σ (yᵢ − f̂(xᵢ))²
```

This measures how closely the model fits the training data.

Important property: Training error always decreases as model flexibility increases.

However, training error is not the true objective.

### Test MSE (Generalization Error)

```text
Test MSE = E[(Y₀ − f̂(X₀))²]
```

This represents expected error on new, unseen data. It is the quantity we truly want to minimize.

---

### Reducible vs Irreducible Error

Expected test MSE can be decomposed as:

```text
E[(Y − f̂(X))²] = [f(X) − f̂(X)]² + Var(ε)
```

Components:

1. Reducible error → error from imperfect estimation of f
2. Irreducible error → Var(ε), inherent noise

Irreducible error represents the lowest achievable prediction error. No model can eliminate randomness.

---

### Bias–Variance Decomposition (Core Concept)

For a specific point x₀:

```text
E[(y₀ − f̂(x₀))²] = Var(f̂(x₀)) + Bias(f̂(x₀))² + Var(ε)
```

This decomposition explains why test error behaves differently from training error.

#### Bias

- Error from simplifying assumptions
- High bias → underfitting
- Model systematically misses structure

More flexibility → bias decreases.

#### Variance

- Sensitivity to fluctuations in training data
- High variance → overfitting
- Model captures noise rather than signal

More flexibility → variance increases.

#### Bias–Variance Tradeoff

As model flexibility increases:

- Bias ↓
- Variance ↑
- Training error ↓
- Test error often follows a U‑shaped curve

The optimal model balances bias and variance to minimize test error.

---

## 2.2.2 Classification Setting

In classification, performance is measured by error rate.

### Error Rate

Error Rate = fraction of observations misclassified

We distinguish:

- Training error rate
- Test error rate

As with regression, training error decreases with flexibility, but test error may increase due to overfitting.

---

### Bayes Classifier

The optimal classifier assigns each point to the class with highest conditional probability:

```text
argmax_j P(Y = j | X = x₀)
```

The error achieved by this classifier is called the Bayes error rate.

The Bayes error rate represents the lowest possible classification error and reflects irreducible uncertainty when class distributions overlap.

---

### K‑Nearest Neighbors (KNN)

KNN estimates:

```text
P(Y = j | X = x₀)
```

by computing the fraction of class j among the K nearest neighbors of x₀.

Small K:

- Very flexible
- Low bias
- High variance
- Likely to overfit

Large K:

- Less flexible
- Higher bias
- Lower variance

The choice of K directly controls the bias–variance tradeoff.

---

# 2.3 Lab: Introduction to Python

This section introduces essential Python tools required for implementing statistical learning methods.

---

## 2.3.1 NumPy

```python
import numpy as np
```

NumPy provides efficient array operations and linear algebra tools.

Essential operations:

- Creating arrays:

```python
np.array([1, 2, 3])
```

- Inspecting shape:

```python
X.shape
```

- Slicing arrays:

```python
X[:, j]
X[i, :]
```

- Vectorized computations:

```python
np.mean((y - yhat)**2)
```

- Matrix multiplication:

```python
X @ beta
```

- Random number generation with reproducibility:

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
```

NumPy arrays are the foundation of nearly all machine learning libraries.

---

## 2.3.2 Visualization with Matplotlib

```python
import matplotlib.pyplot as plt
```

Common usage:

```python
plt.scatter(X[:, 0], y)
plt.plot(x_vals, y_vals)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Visualization is critical for:

- Exploring data distributions
- Detecting outliers
- Understanding relationships
- Diagnosing model behavior

---

## 2.3.3 Basic scikit‑learn Workflow

Standard supervised ML pattern:

1. Split data into training and testing sets
2. Fit model on training data
3. Generate predictions
4. Evaluate performance

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
```

This workflow pattern is repeated throughout the book and forms the backbone of applied machine learning projects.

---

