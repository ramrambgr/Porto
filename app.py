import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load data
@st.cache
def load_data():
    df = pd.read_csv("train.csv")
    return df

df1 = load_data()

# App Title
st.title("Academic Success Prediction")

# Sidebar
st.sidebar.header("User Input Features")
st.sidebar.markdown("Use the checkboxes below to explore the dataset and model performance")

# Show the dataset
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df1.head())

# Dataset Information
if st.sidebar.checkbox("Show dataset info"):
    st.subheader("Dataset Information")
    st.write(df1.shape)
    st.write(df1.info())
    st.write(df1.describe())

# Handling missing values, duplicates, and categorical encoding
df1['Target'] = df1['Target'].replace({'Graduate': 0, 'Dropout': 1, 'Enrolled': 2})

# Show bar plot for target variable distribution
if st.sidebar.checkbox("Show target distribution"):
    st.subheader("Distribution of Academic Success")
    fig, ax = plt.subplots()
    sns.countplot(data=df1, x='Target', edgecolor='black', ax=ax)
    ax.set_title('Distribution of Academic Success')
    st.pyplot(fig)

# Show boxplots for numerical features
if st.sidebar.checkbox("Show boxplots for numerical features"):
    st.subheader("Boxplots for Numerical Features")
    numerical_features = ['Admission grade', 'Age at enrollment', 'Previous qualification (grade)',
                          'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
                          'Unemployment rate', 'Inflation rate', 'GDP']
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, feature in enumerate(numerical_features):
        sns.boxplot(data=df1, y=feature, ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(f'{feature}')
    plt.tight_layout()
    st.pyplot(fig)

# Show countplots for categorical features
if st.sidebar.checkbox("Show countplots for categorical features"):
    st.subheader("Countplots for Categorical Features")
    categorical_features = ['Marital status', 'Application mode', 'Daytime/evening attendance',
                            'Previous qualification', 'Gender', 'Scholarship holder']
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for i, feature in enumerate(categorical_features):
        sns.countplot(data=df1, x=feature, hue='Target', ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(f'{feature}')
    plt.tight_layout()
    st.pyplot(fig)

# Train a logistic regression model
X = df1.drop(columns=['Target', 'id'])
y = df1['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.95, random_state=2)
logit = LogisticRegression(multi_class='ovr')
logit.fit(X_train, y_train)
y_pred_logit = logit.predict(X_test)

# Display model performance
if st.sidebar.checkbox("Show model performance"):
    st.subheader("Logistic Regression Performance")
    score = accuracy_score(y_pred_logit, y_test)
    st.write(f"Accuracy Score of Logistic Regression: {score*100:.2f}%")
    st.write(f"Error of Logistic Regression: {(1 - score)*100:.2f}%")

    st.subheader("Classification Report")
    st.text(classification_report(y_pred_logit, y_test))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_pred_logit, y_test), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Run the app
if __name__ == '__main__':
    st.write('Streamlit app is running')
