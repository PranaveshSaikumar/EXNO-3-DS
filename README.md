## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Name: Pranavesh Saikumar
Reg No: 212223040149
```
```
import pandas as pd
df = pd.read_csv("C:/Users/admin/Documents/DS files/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/fe179f66-0ca3-41d2-9ffc-8cc104633201)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/396487ca-0eb4-4cd1-b085-469ac3c4ce85)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/8b536e94-680f-4f0d-b917-82dce9765a20)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/48502635-7e3a-4999-84de-628466b0ca4d)

```
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)  
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/bee2c223-4e9c-4a8e-aa6c-9d3ba21fea96)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/3f4bd79f-d5d7-4090-8f4a-708b3cf814dc)

```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/d06753ab-fa97-4314-b7f2-fc5f1f068f8a)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/ebc16903-b652-46a5-9892-5042d88bcade)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/2d2fa8ed-36bc-44a2-af3f-f9be38232b6c)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/93b9812e-9ace-493b-9487-2f4d561b3ca1)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/b8552a2d-43ba-4465-b64b-08c9f693ece7)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/132c7bae-0fe9-4082-bf85-0728c7782fab)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/d6532be7-de61-4f7e-961f-d04c356f981b)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/efbe9313-0432-4e69-ba72-80acdc2bde08)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7c7395ae-1d85-464e-b3dc-c7665f88b148)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/8de3c684-e120-4315-818d-76d78e8e2011)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/10ff86ea-49b7-4836-99f0-8508797086ea)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/fb8f8fe9-c0f5-46c1-b18d-29017906a40b)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/b6cf2354-1658-424f-b030-c393184dfe1a)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5fd437da-b98c-45ac-9885-590a64a8038f)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e2c50b49-e929-449c-92cf-50e40169f616)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/80413849-b550-43a5-bcf7-17ff802b1e0f)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/85706c9c-5a8a-440f-b64d-96c7490975d1)

```
dt = pd.read_csv("C:/Users/admin/Documents/DS files/titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/c0c3f1d2-ce71-460a-bf50-d32dfe771f7a)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/c171623e-ec81-4844-9241-d2a21418000c)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3e293a84-d32d-4e41-8cb0-f07e1798777b)



# RESULT:
Thus, the given data was read successfully, and Feature Encoding and Transformation processes were performed successfully.
       

       
