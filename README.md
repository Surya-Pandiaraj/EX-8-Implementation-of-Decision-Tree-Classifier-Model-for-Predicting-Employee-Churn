### NAME: SURYA P <br>
### REG NO: 212224230280

# EX 8 : IMPLEMENTATION OF DECISION TREE CLASSIFIER MODEL FOR PREDICTING EMPLOYEE CHURN

## AIM :

To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn

## EQUIPMENTS REQUIRED :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :

1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## PROGRAM :

```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SURYA P
RegisterNumber:  212224230280

*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("C:\\Users\\admin\\Desktop\\Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x. head () #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## OUTPUT :

![image](https://github.com/user-attachments/assets/eb3fcbff-f863-4803-a50c-97fb9c66b87d)

![image](https://github.com/user-attachments/assets/f29fe447-256f-418f-a13f-c65a24fbcdcf)

![image](https://github.com/user-attachments/assets/028c8d2c-a09f-4613-a726-455921ce32fd)

![image](https://github.com/user-attachments/assets/9f3daa78-fc38-4b86-991e-6023640dc098)

## RESULT :
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
