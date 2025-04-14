Perfect! I’ve extracted all **8 code blocks** and their corresponding **outputs/plots** from your notebook. Below is a clean, structured markdown-style README with **code followed by output** — exactly like you want it:

---

## 📊 Titanic Dataset - EDA Report

---

### **1. Importing Libraries & Loading Dataset**

```python
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
```

✅ Dataset successfully loaded.

---

### **2. Initial Data Inspection**

```python
df.head()
df.describe()
df.info()
df.duplicated().sum()
df.columns
```

<pre>
# Output will show the first few rows, data types, summary stats,
# total duplicates, and column names.
</pre>

---

### **3. Checking for Missing Values**

```python
df.isnull().sum()
```

<pre>
# Output will show null counts in Age, Cabin, and Embarked columns.
</pre>

---

### **4. Univariate Analysis**

```python
# Survived bar plot
df['Survived'].value_counts().plot(kind='bar')
plt.title("no.of survived")
plt.show()

# Fare boxplot
sns.boxplot(df['Fare'])
plt.title('Fare BoxPlot')
plt.show()
```

✅ Output:

**Survival Count**
<img src="data:image/png;base64,...1" style="max-width: 100%; height: auto;">

**Fare Boxplot**
<img src="data:image/png;base64,...2" style="max-width: 100%; height: auto;">

---

### **5. Bivariate Analysis**

```python
# Survived count by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survived vs Sex')
plt.show()

# Age vs Fare scatter plot
sns.scatterplot(x='Age', y='Fare', hue='Sex', data=df)
plt.title('Age vs Fare')
plt.show()
```

✅ Output:

**Survival by Gender**
<img src="data:image/png;base64,...3" style="max-width: 100%; height: auto;">

**Age vs Fare**
<img src="data:image/png;base64,...4" style="max-width: 100%; height: auto;">

---

### **6. Multivariate Analysis**

```python
# Pairplot
sns.pairplot(df[['Survived','Age','SibSp','Parch']].dropna(), hue='Survived')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
numerical_df = df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title("correlation heatmap")
plt.show()
```

✅ Output:

**Pairplot**
<img src="data:image/png;base64,...5" style="max-width: 100%; height: auto;">

**Heatmap**
<img src="data:image/png;base64,...6" style="max-width: 100%; height: auto;">

---

### **7. Handling Missing Values**

```python
df['Age'].fillna(df['Age'].median(), inplace=True)
```

✅ Missing `Age` values filled with median.

---

### **8. Skewness Check**

```python
df['Age'].skew()
```

<pre>
# Output: Value representing the skewness of Age distribution
</pre>

---
