# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Data Collection
import pandas as pd
ds=pd.read_csv(r'C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\titanic sarvaival ml 4project\titanic.csv')
print(ds)
print(ds.head(5))
print(ds.tail(5))
print(ds.info)
print(ds.shape)
print(ds.dtypes)
print(ds.describe)


# # Step 2: Data Cleaning and Preprocessing

#finding missing value

print(ds.isnull().sum())
# # Replace any missing values with the mean value of the column
ds.fillna(ds.mean(), inplace=True)

ds['Age']= ds['Age'].fillna(0)
ds.isnull().sum()

#ds['Embarked']=ds['Embarked'].fillna(ds['Embarked'].m())
ds['Embarked']= ds['Embarked'].fillna(0)
#replace data string to numeric

ds = ds.replace({'Sex': {'male': 0, 'female': 1},
                  'Embarked': {'S': 0, 'C': 1, 'Q': 2}})
print(ds.head(5))
#value count  for graph
print(ds['PassengerId'].value_counts())
print(ds['Survived'].value_counts())
print(ds['Pclass'].value_counts())
print(ds['Age'].value_counts())
print(ds['Ticket'].value_counts())
print(ds['Name'].value_counts())
print(ds['SibSp'].value_counts())
print(ds['Parch'].value_counts())
print(ds['Fare'].value_counts())
print(ds['Embarked'].value_counts())
print(ds['Cabin'].value_counts())
print(ds['Sex'].value_counts())



print(ds.isnull().sum())
#make a graph 
#visualilze the  all data

import matplotlib.pyplot as plt
import seaborn as sns



Cabin_data =ds['Cabin'].value_counts()
Embarked_data =ds['Embarked'].value_counts()
Fare_data=ds['Fare'].value_counts()
PassengerId_data=ds['PassengerId'].value_counts()
Survived_data=ds['Survived'].value_counts()
Pclass_data=ds['Pclass'].value_counts()
Age_data=ds['Age'].value_counts()
Ticket_data=ds['Ticket'].value_counts()
Name_data=ds['Name'].value_counts()
SibSp_data=ds['SibSp'].value_counts()
Parch_data=ds['Parch'].value_counts()
Sex_data=ds['Sex'].value_counts()

Pclass_label=['1','2','3']
Survived_label=['o','1']
SibSp_label=['0','1','2','3','4','5','8']
Parch_label=['0','1','2','3','4','5','6']
Embarked_label=['S','C','Q']
Sex_label=['male','female']




plt.figure(figsize=(15,10))
plt.suptitle("Distibution analysis")
plt.subplot(3,3,1)
plt.plot(Sex_label,Sex_data,color='red')
plt.grid(True)
plt.title('sex_graph')

plt.subplot(3,3,2)
plt.scatter(x=Embarked_label,y=Embarked_data,color='green')
plt.grid(True)
plt.title('Embarke_graph')

plt.subplot(3,3,3)
plt.scatter(x=SibSp_label,y=SibSp_data)
plt.grid(True)
plt.title('SibSp_graph')

plt.subplot(3,3,4)
plt.scatter(x=Survived_label ,y=Survived_data, color='red')
plt.grid(True)
plt.title('Survived_graph')

plt.subplot(3,3,5)
plt.bar(Pclass_label,Pclass_data)
plt.grid(True)
plt.title('Pclass_graph')

plt.show()


#droping
ds=ds.drop('Name',axis='columns')
ds=ds.drop('Ticket',axis='columns')

ds=ds.drop('Cabin',axis='columns')
ds=ds.drop('SibSp',axis='columns')
ds=ds.drop('Parch',axis='columns')
ds=ds.drop('Fare',axis='columns')

#find x and y

x = ds.drop(['Survived'], axis=1)
y = ds['Survived'].values
print(x)
print(y)

#testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# preprocessing
from sklearn .preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_train)

print(ds.isnull().sum())


#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_1=LogisticRegression()
model_1.fit(x_train,y_train)

model_2 = DecisionTreeClassifier()
model_2.fit(x_train, y_train)

model_3 = RandomForestClassifier()
model_3.fit(x_train, y_train)

#predict of model

pred_1=model_1.predict(x_test)
pred_2 = model_2.predict(x_test)
pred_3 =model_3.predict(x_test)
##if new passenger want to traval then
Passenger=(input("enter name:-"))
Passengerid=(input("enter Passengerid:-"))
Pclass=int(input('enter pclass no:-'))
gender=int(input('enter gender if male=1 else=0 :-'))
Age=int(input('enter Age:-'))
Embarked=int(input('enter Embarked S: 0, C: 1, Q: 2:-'))
newperson=[[Passengerid,Pclass,gender,Age,Embarked]]
result=model_3.predict(sc.transform(newperson))
print('------------------------------------------------------------')
print(result)
if  result==1:
    print("person is survived")
else:
    print('person is not survived')
print('------------------------------------------------------------')
#output
 # Calculate the accuracy of each model
print("accuracy_score of 1st model:-{0}%".format(accuracy_score(y_test,pred_1)*100))
print("accuracy_score of 2st model:-{0}%".format(accuracy_score(y_test,pred_2)*100))
print("accuracy_score of 3st model:-{0}%".format(accuracy_score(y_test,pred_3)*100))

# Calculate the confusion matrix for each model
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_1)))
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_2)))
print("confusion_matrix of 1st model:-{0}%".format(confusion_matrix(y_test,pred_3)))