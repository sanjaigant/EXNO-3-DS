## EXNO-3-DS
## Developed by : sanjai ganth.B
## Reg No : 212224230244
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
```
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
```
  # 2. POWER TRANSFORMATION
```
• Boxcox method
• Yeojohnson method
```

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="353" height="446" alt="image" src="https://github.com/user-attachments/assets/7b69519e-ea12-4303-8387-1f1d21f0adb0" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="252" height="266" alt="image" src="https://github.com/user-attachments/assets/4a16ed56-09e5-4643-a2db-8f4d1afec205" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="393" height="452" alt="image" src="https://github.com/user-attachments/assets/f1bdc0f4-945d-40eb-b0e0-3012434dedd9" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="818" height="441" alt="image" src="https://github.com/user-attachments/assets/4eae0612-d9a6-482d-837d-83ede587baf0" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="1027" height="679" alt="image" src="https://github.com/user-attachments/assets/474dae7e-bfb7-4d9c-a8ea-c695fe13084c" />

```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="520" height="441" alt="image" src="https://github.com/user-attachments/assets/0102c0d9-e165-4e5c-a07c-1ea80e7352ad" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["nom_0"],y=CC["ord_2"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="442" height="459" alt="image" src="https://github.com/user-attachments/assets/e7cfdd99-c1bd-4cc0-ac1c-3ddb94c32d8b" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="829" height="548" alt="image" src="https://github.com/user-attachments/assets/e5225d94-5005-44bf-ac7b-0b401adfecef" />

```
df.skew()
```

<img width="402" height="261" alt="image" src="https://github.com/user-attachments/assets/486161ed-80b0-4498-a64e-60ca5a88d4f5" />

```
np.log(df["Highly Positive Skew"])
```

<img width="383" height="575" alt="image" src="https://github.com/user-attachments/assets/a7264de8-3c16-47a4-9364-8e06f5af92eb" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="382" height="580" alt="image" src="https://github.com/user-attachments/assets/2e7b7219-5038-41df-b616-c6e064a4a1de" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="365" height="573" alt="image" src="https://github.com/user-attachments/assets/a9215d2c-85a9-4203-8ffc-6a66bfe48fe3" />

```
np.square(df["Highly Positive Skew"])
```

<img width="415" height="584" alt="image" src="https://github.com/user-attachments/assets/57dc96cc-2f75-4c7f-b8f3-a87bc5187642" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="889" height="630" alt="image" src="https://github.com/user-attachments/assets/32152857-4851-4eeb-b0da-38396b1b3b7f" />

```
df.skew()
```

<img width="472" height="278" alt="image" src="https://github.com/user-attachments/assets/60ceaab0-1315-480f-a3cf-a3e1694a599c" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="499" height="310" alt="image" src="https://github.com/user-attachments/assets/fff9dca6-3e5b-4810-9865-7a97541a6c8e" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="839" height="554" alt="image" src="https://github.com/user-attachments/assets/c966dad7-6c19-48d5-b919-b05411daa5e1" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="753" height="582" alt="image" src="https://github.com/user-attachments/assets/7a0f4d95-07ea-40e5-bf7d-d99ced8f5958" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="734" height="558" alt="image" src="https://github.com/user-attachments/assets/d27d8b3e-899d-4cb8-8519-6a0e021bdcd1" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="746" height="575" alt="image" src="https://github.com/user-attachments/assets/5749af5f-3ad8-4241-989b-c8381d0e224d" />


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="748" height="548" alt="image" src="https://github.com/user-attachments/assets/e3a1031a-1c8a-4ae7-8afa-b875aeb68b31" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

<img width="720" height="552" alt="image" src="https://github.com/user-attachments/assets/941e862f-3682-4441-84a3-1fba71841ef7" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="768" height="562" alt="image" src="https://github.com/user-attachments/assets/2d7a9cf5-ed82-431f-a305-be4af9c76e80" />

# RESULT:
        Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
