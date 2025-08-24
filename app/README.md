# Build_Classification_Models_From_Scratch


# Classification Models in Machine Learning 📊

This README provides a simple yet complete explanation of major classification algorithms used in Machine Learning.  
Each model includes:
- 📌 Explanation
- 🔢 Formula / Concept
- 🐍 Python Example
- ⚖️ Pros & Cons
- 🌍 Applications

---

## 1. Logistic Regression

### 📌 Explanation
A linear model for **binary classification**. It predicts the probability of belonging to a class using the **sigmoid function**.

### 🔢 Formula
z = β0 + β1x1 + β2x2 + ... + βnxn  
P(y=1|x) = 1 / (1 + e^(-z))

### 🐍 Python Example
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Simple, interpretable, fast  
❌ Works only for linear boundaries  

### 🌍 Applications
- Spam detection  
- Disease diagnosis  

---

## 2. K-Nearest Neighbors (KNN)

### 📌 Explanation
A **non-parametric algorithm** that classifies based on the **majority class of k nearest neighbors**.

### 🔢 Formula
Distance: Euclidean  
d(x,y) = √Σ(xi - yi)^2  

### 🐍 Python Example
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Simple, no training phase  
❌ Slow for large datasets, sensitive to noisy data  

### 🌍 Applications
- Recommender systems  
- Image recognition  

---

## 3. Decision Tree

### 📌 Explanation
A **tree-based model** that splits features based on criteria like **Gini Impurity** or **Entropy**.

### 🔢 Formula
Entropy = - Σ p * log2(p)  

### 🐍 Python Example
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Easy to interpret, handles non-linear data  
❌ Prone to overfitting  

### 🌍 Applications
- Customer churn prediction  
- Loan approval  

---

## 4. Random Forest

### 📌 Explanation
An **ensemble of decision trees** using **bagging** to reduce overfitting and improve accuracy.

### 🐍 Python Example
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ High accuracy, robust to noise  
❌ Slower, less interpretable  

### 🌍 Applications
- Fraud detection  
- Healthcare predictions  

---

## 5. Support Vector Machine (SVM)

### 📌 Explanation
Finds the **optimal hyperplane** that maximizes the margin between classes.

### 🔢 Formula
f(x) = w·x + b  
Decision boundary: w·x + b = 0  

### 🐍 Python Example
```python
from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Works well on high-dimensional data  
❌ Training can be slow for large datasets  

### 🌍 Applications
- Face detection  
- Text classification  

---

## 6. Naive Bayes

### 📌 Explanation
A **probabilistic classifier** based on **Bayes' theorem**, assuming independence between features.

### 🔢 Formula
P(y|x) = [P(x|y) * P(y)] / P(x)  

### 🐍 Python Example
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Fast, works well with text data  
❌ Assumes feature independence (not always true)  

### 🌍 Applications
- Email spam filtering  
- Sentiment analysis  

---

## 7. Gradient Boosting (XGBoost, LightGBM)

### 📌 Explanation
An **ensemble boosting algorithm** that builds trees sequentially, each correcting the previous one.

### 🔢 Formula
New model = Previous model + Learning rate * Weak learner  

### 🐍 Python Example
```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Very powerful, high accuracy  
❌ Computationally expensive  

### 🌍 Applications
- Kaggle competitions 🏆  
- Credit scoring  

---

## 8. Neural Networks (Basic)

### 📌 Explanation
A network of **neurons (nodes)** organized in layers. Each neuron applies a weighted sum + activation function.

### 🔢 Formula
z = Σ(wi·xi + b)  
a = σ(z) (e.g., sigmoid, ReLU)  

### 🐍 Python Example
```python
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ⚖️ Pros & Cons
✅ Handles complex non-linear data  
❌ Requires large data, hard to interpret  

### 🌍 Applications
- Speech recognition  
- Image classification  

---

# 📌 Summary Table

| Model              | Strengths                   | Weaknesses               |
|--------------------|-----------------------------|--------------------------|
| Logistic Regression| Simple, interpretable       | Linear boundaries only   |
| KNN                | Easy, no training           | Slow on big data         |
| Decision Tree      | Intuitive, interpretable    | Overfitting              |
| Random Forest      | Robust, high accuracy       | Less interpretable       |
| SVM                | Good on high-dim data       | Slow for large datasets  |
| Naive Bayes        | Fast, good for text         | Independence assumption  |
| Gradient Boosting  | Very accurate               | Computationally heavy    |
| Neural Networks    | Handles complex patterns    | Needs large data & compute|

---

📖 This README serves as a **cheat sheet for classification models** in ML.  
