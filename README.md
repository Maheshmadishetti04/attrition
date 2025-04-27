# attrition


# streamlit_app.py

# Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

# Streamlit App
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")
st.title("Employee Attrition Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/madishettimahesh/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

attrition_df = load_data()

st.subheader("First Look at the Data")
st.dataframe(attrition_df.head())

# Data Cleaning Note
st.info("✅ Data has **zero NaN values**.")

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

# --- Attrition Distribution
st.subheader("Attrition Count Plot")
attrition_counts = attrition_df['Attrition'].value_counts()
colors = ['skyblue', 'lightcoral']

fig1, ax1 = plt.subplots()
ax1.bar(attrition_counts.index, attrition_counts.values, color=colors)
ax1.set_title('Attrition Count')
ax1.set_xlabel('Attrition')
ax1.set_ylabel('Number of Employees')
st.pyplot(fig1)

st.markdown("""
### Attrition Distribution Insights:
- A significant majority of employees have **not left** the company.
- A smaller proportion have experienced **attrition**.
""")

# --- Age Distribution
st.subheader("Age Distribution of Employees")

fig2, ax2 = plt.subplots(figsize=(8,7))
sns.histplot(attrition_df['Age'], bins=10, edgecolor='black', color='blue', kde=True, ax=ax2)
ax2.set_title('Age Distribution of Employees')
ax2.set_xlabel('Age')
ax2.set_ylabel('Number of Employees')
st.pyplot(fig2)

st.markdown("""
### Age Distribution Insights:
- The age distribution appears relatively **balanced** but **slightly skewed** towards the younger demographic.
- Majority of employees are between **30-40 years old**.
- Suggests a **relatively young workforce** impacting attrition, career progression, and training needs.
""")

# --- Attrition vs Job Role
st.subheader("Attrition by Job Role")

fig3, ax3 = plt.subplots(figsize=(12,6))
sns.countplot(x='JobRole', hue='Attrition', data=attrition_df, palette='coolwarm', ax=ax3)
ax3.set_title('Attrition by Job Role')
ax3.set_xlabel('Job Role')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

st.markdown("""
### Attrition by Job Role Insights:
- Roles like **Sales Representatives** and **Laboratory Technicians** have higher attrition rates.
- Senior roles like **Research Directors** and **Managers** show **lower attrition**, suggesting greater stability at higher levels.
""")

# --- Attrition vs OverTime
st.subheader("Attrition by Overtime Status")

fig4, ax4 = plt.subplots(figsize=(8,5))
sns.countplot(x='OverTime', hue='Attrition', data=attrition_df, palette='coolwarm', ax=ax4)
ax4.set_title('Attrition by Overtime Status')
st.pyplot(fig4)

st.markdown("""
### Attrition by Overtime Insights:
- Employees working **OverTime** are more likely to experience **attrition**.
- Indicates **dissatisfaction** or **burnout** could be a key reason for leaving.
""")

# --- Attrition vs Work-Life Balance
st.subheader("Attrition by Work-Life Balance")

fig5, ax5 = plt.subplots(figsize=(8,5))
sns.countplot(x='WorkLifeBalance', hue='Attrition', data=attrition_df, palette='coolwarm', ax=ax5)
ax5.set_title('Attrition by Work-Life Balance')
st.pyplot(fig5)

st.markdown("""
### Attrition by Work-Life Balance Insights:
- Lower **Work-Life Balance** (rated 1 or 2) is strongly associated with **higher attrition**.
- Better balance (rated 3 or 4) helps in **retaining employees**.
""")

# --- Attrition vs Education
st.subheader("Attrition by Education Level")

fig6, ax6 = plt.subplots(figsize=(8,5))
sns.countplot(data=attrition_df, x='Education', hue='Attrition', palette='coolwarm', ax=ax6)
ax6.set_title('Attrition by Education Level')
ax6.set_xlabel('Education Level')
ax6.set_ylabel('Count')
st.pyplot(fig6)

st.markdown("""
### Attrition by Education Insights:
- **Higher attrition** observed among employees with a **Bachelor's Degree**.
""")

# --- Monthly Income vs Attrition
st.subheader("Monthly Income vs Attrition Status")

fig7, ax7 = plt.subplots(figsize=(8,5))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=attrition_df, palette='Set2', ax=ax7)
ax7.set_title('Monthly Income vs Attrition Status')
ax7.set_xlabel('Attrition')
ax7.set_ylabel('Monthly Income')
st.pyplot(fig7)

# --- Correlation Heatmap
st.subheader("Correlation Heatmap of Numerical Features")

numeric_features = attrition_df.select_dtypes(include=['int64', 'float64'])

fig8, ax8 = plt.subplots(figsize=(12,10))
sns.heatmap(numeric_features.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax8)
ax8.set_title('Correlation Heatmap of Numerical Features')
st.pyplot(fig8)

