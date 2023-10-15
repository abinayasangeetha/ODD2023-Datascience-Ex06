# EX-06 FEATURE TRANSFORMATION
### Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
### Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
### Program:
```
Developed By: ABINAYA S
Register No: 212222230002
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## OUTPUT:
### Original Data:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/3d1d164e-c31d-4a8e-ba7e-6a94582fbb28)

### Data information:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/2ab91ba5-f025-4466-9dce-5ee22b2e0ccb)

### Data Describe:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/0f58e2bf-9d72-4d70-bf4a-41ff50ae9d47)

### Before transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/8c6ac148-7d54-413c-b231-0c6d2aa08f97)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/e4e9abac-6932-4a6e-9370-e046beb629cc)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/07e848f3-016a-4dfb-99a8-6f66b2962b6b)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/c0f633cb-3c02-4439-a3f4-18ae1d0755a5)

### Log transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/078f77f2-7566-41de-a8db-9c68c98ca1cf)

### Reciprocal transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/89a33717-793f-4cc0-8226-d21682d14337)

### Square root transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/300f2e17-56be-48b7-a3bf-82fa13ac9921)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/7fd9de8f-5b18-464e-a829-0f3097d848f2)

### Power transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/605afa0c-ec77-494c-af6e-c59ed6e5b24c)
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/9ac080a9-f87c-481f-973f-e2ba95150ee5)

### Quantile transformation:
![image](https://github.com/abinayasangeetha/ODD2023-Datascience-Ex06/assets/119393675/ae03f73c-b9c4-4166-a6b6-8d2550a1ab9a)

### Result:  
Thus feature transformation is done for the given dataset.
