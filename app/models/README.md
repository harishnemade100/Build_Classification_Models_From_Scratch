import pypandoc

# Full README content
readme_content = """
# Logistic Regression 📊  

Logistic Regression is a **supervised learning algorithm** used for **binary classification** problems (yes/no, 0/1, true/false). Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts **probabilities** that a given input belongs to a class.

---

## 1. Formula 🔢  

### 1.1 Linear Equation
At first, logistic regression creates a linear combination of input features:

\\[
z = \\beta_0 + \\beta_1x_1 + \\beta_2x_2 + \\dots + \\beta_nx_n
\\]

Where:  
- \\( z \\) = linear score  
- \\( \\beta_0 \\) = intercept (bias term)  
- \\( \\beta_i \\) = coefficients for features  
- \\( x_i \\) = input features  

### 1.2 Sigmoid Function
Since the output of the linear equation can be any real number, we map it into a **probability** between 0 and 1 using the **sigmoid function**:

\\[
P(y=1|x) = \\sigma(z) = \\frac{1}{1 + e^{-z}}
\\]

- If \\( P(y=1|x) > 0.5 \\) → Predict **1**  
- If \\( P(y=1|x) \\leq 0.5 \\) → Predict **0**  

---

## 2. Cost Function ⚖️  

Logistic Regression uses **Log Loss (Cross-Entropy Loss)** instead of Mean Squared Error:

\\[
J(\\beta) = -\\frac{1}{m}\\sum_{i=1}^m \\Big[ y^{(i)}\\log(\\hat{y}^{(i)}) + (1-y^{(i)})\\log(1-\\hat{y}^{(i)}) \\Big]
\\]

Where:  
- \\( m \\) = number of samples  
- \\( y^{(i)} \\) = actual label  
- \\( \\hat{y}^{(i)} \\) = predicted probability  

Goal: **Minimize cost function** using optimization algorithms (Gradient Descent).

---

## 3. Gradient Descent 🔄  

Update rule for each parameter:

\\[
\\beta_j := \\beta_j - \\alpha \\frac{\\partial J(\\beta)}{\\partial \\beta_j}
\\]

Where:  
- \\( \\alpha \\) = learning rate  
- \\( \\frac{\\partial J(\\beta)}{\\partial \\beta_j} \\) = derivative of cost function w.r.t parameter  

---

## 4. Example: Predicting Student Admission 🎓  

### Dataset (Simplified)
| Hours Studied (x1) | Previous Score (x2) | Admitted (y) |
|---------------------|----------------------|--------------|
| 2                   | 50                   | 0            |
| 4                   | 60                   | 0            |
| 6                   | 70                   | 1            |
| 8                   | 85                   | 1            |
| 10                  | 90                   | 1            |

---

### Step 1: Linear Combination
\\[
z = \\beta_0 + \\beta_1 \\times (Hours) + \\beta_2 \\times (Score)
\\]

### Step 2: Sigmoid
\\[
P(Admitted=1|x) = \\frac{1}{1 + e^{-z}}
\\]

---

## 5. Python Implementation 🐍  

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample dataset
data = {
    "Hours": [2, 4, 6, 8, 10],
    "Score": [50, 60, 70, 85, 90],
    "Admitted": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features & target
X = df[["Hours", "Score"]]
y = df["Admitted"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict probability for a new student
new_student = [[7, 75]]
prob = model.predict_proba(new_student)[0][1]
pred = model.predict(new_student)

print(f"Probability of Admission: {prob:.2f}")
print(f"Predicted Class: {pred[0]}")
