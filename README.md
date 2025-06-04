# 🏠 House Price Prediction using Linear Regression from Scratch

This project demonstrates how to build a simple **Linear Regression** model **from scratch** in Python using **NumPy** and **Matplotlib**, without using any machine learning libraries. The model is trained using **Gradient Descent** to predict **house prices** based on the **size** of the house.

---

## 📁 Dataset

The dataset used is a simple CSV/txt file with the following columns:

- `size`: Size of the house (e.g., in square feet or square meters)
- `price`: Price of the house

> Example row:  
> `2104, 399900`

### 📌 File Used:
house-price-prediction-using-linear-regression.txt
---

## 📊 Objective

- Implement Linear Regression using only NumPy (no sklearn or ML libraries)
- Use Gradient Descent to optimize parameters (weights and bias)
- Visualize the loss (RSS), learned regression line, and predictions vs actual values

---

**Linear Regression Formula**  
ŷ = b₀ + w₁ · x  

Where:  
- ŷ is the predicted price  
- x is the house size (input feature)  
- w₁ is the weight (slope)  
- b₀ is the bias (intercept)

---

## 🚀 How It Works

### 1. **Preprocessing**
- Input features and targets are **standardized** (scaled to mean 0, std 1)

### 2. **Initialize Parameters**
- Start with `w1 = 0`, `b0 = 0`
- Define learning rate and number of epochs

### 3. **Training**
- For each epoch:
  - Predict prices using current weights
  - Calculate errors
  - Compute gradients
  - Update parameters using gradient descent

### 4. **Evaluation**
- Plot RSS over epochs
- Plot learned regression line
- Plot predictions vs actual values

---
