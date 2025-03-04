# CustomerChurn
# Churn Prediction Model & Streamlit App

## 📌 Project Overview

This project builds a **Customer Churn Prediction Model** using multiple machine learning algorithms and deploys it as a **Streamlit web application**. The app allows users to input customer details and predict the likelihood of churn.

---

## 🗂 Dataset Description

The dataset contains customer information and telecom usage statistics, with a target column **"Churn"** indicating whether a customer left the service (1) or stayed (0).

### **Features in the dataset:**

- **Customer Account Information:**
  - `State`: The U.S. state of the customer
  - `Account length`: Duration of customer relationship (in months)
  - `International plan`: Whether the customer has an international plan (Yes/No)
  - `Voice mail plan`: Whether the customer has a voice mail plan (Yes/No)
- **Usage Statistics:**
  - `Total day minutes`, `Total day calls`, `Total day charge`
  - `Total evening minutes`, `Total evening calls`, `Total evening charge`
  - `Total night minutes`, `Total night calls`, `Total night charge`
  - `Total intl minutes`, `Total intl calls`, `Total intl charge`
- **Customer Support Interactions:**
  - `Customer service calls`: Number of times the customer called customer service

The goal is to predict **Churn** (1 = Churn, 0 = No Churn) based on the above features.

---

## 🚀 Project Workflow

1. **Exploratory Data Analysis (EDA)** – Data cleaning, missing value analysis, correlation analysis, visualizations.
2. **Data Preprocessing** – Encoding categorical variables, scaling numerical features, feature selection.
3. **Model Training & Evaluation** – Training multiple models (Logistic Regression, SVM, Random Forest, Gradient Boosting, Neural Networks) and selecting the best.
4. **Saving the Best Model** – Saving the trained model and scalers using pickle.
5. **Streamlit App Development** – Building a web app for users to interact with the model.
6. **Deployment** – Running the app locally or deploying it to a cloud platform like **Streamlit Cloud, Heroku, or AWS**.

---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**

```sh
git clone https://github.com/your-username/churn-prediction-streamlit.git
cd churn-prediction-streamlit
```

### **2️⃣ Create Virtual Environment & Install Dependencies**

```sh
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**

```sh
streamlit run churn_app.py
```

This will start the web app locally. Open the displayed URL in your browser.

---

## 📊 Exploratory Data Analysis (EDA)

The EDA process includes:

- Checking for missing values and data distributions.
- Encoding categorical features (`State`, `International Plan`, `Voice Mail Plan`).
- Generating visualizations:
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Churn distribution
  sns.countplot(x="Churn", data=train_df)
  plt.title("Churn Distribution")
  plt.show()

  # Correlation heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm")
  plt.title("Feature Correlations")
  plt.show()
  ```

---

## 🤖 Machine Learning Models Used

We train the following models and compare their performance:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting**
- **Neural Network (MLP)**

### **Model Training & Evaluation**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on validation data
y_pred = model.predict(X_val_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
```

### **Saving the Best Model**

```python
import pickle

with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
```

---

## 🖥️ Streamlit App Usage

The app allows users to input customer details and predict churn.

### **App Features**

- Input fields for customer details
- "Predict Churn" button
- Displays **Churn Prediction: Yes/No**

### **Run the App**

```sh
streamlit run churn_app.py
```

---

## 🌎 Deployment

### **Deploy on Streamlit Cloud**

1. Push the project to **GitHub**
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy by linking your repository

---

## 📌 Example User Inputs & Output

### **User Inputs**

| Feature                | Value |
| ---------------------- | ----- |
| Account Length         | 120   |
| International Plan     | No    |
| Voice Mail Plan        | Yes   |
| Total Day Minutes      | 250.5 |
| Customer Service Calls | 2     |

### **Predicted Output:**

```
Churn Prediction: No (Customer is likely to stay)
```

---

## 🏆 Results & Key Insights

- **Random Forest** performed the best with \~85% accuracy.
- **Customer Service Calls** and **Total Day Minutes** are strong indicators of churn.
- Customers with **higher customer service calls** are more likely to churn.

---

## 📌 To-Do / Future Improvements

- Deploy on **AWS or Heroku**
- Implement **Feature Importance Analysis**
- Improve UI/UX with enhanced visualizations

---

## 📜 License

This project is **MIT licensed**. Feel free to use and modify it.

---

## 💡 Contributing

Want to improve this project? Fork it and submit a pull request! 🚀

---

## 📬 Contact

For any queries, contact me at [**your-email@example.com**](mailto\:your-email@example.com) or reach out via **GitHub Issues**.

