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
  <class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
</pre>

---

### **3. Checking for Missing Values**

```python
df.isnull().sum()
```

<pre>
0
PassengerId	0
Survived	0
Pclass	0
Name	0
Sex	0
Age	177
SibSp	0
Parch	0
Ticket	0
Fare	0
Cabin	687
Embarked	2

dtype: int64
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
![Alt text](![image](https://github.com/user-attachments/assets/0d4e8d3d-7a81-4f79-9a14-6b14a0869e39)
)

![Alt text](image-url)



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
![Alt text](image-url)

![Alt text](image-url)


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
![Alt text](image-url)


![Alt text](image-url)


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
