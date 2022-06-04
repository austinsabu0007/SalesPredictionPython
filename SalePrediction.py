import pandas as pd #pandas are used in this program to read csv file..this line import pandas
import numpy as np  #numpy is used to do sum operations in array in the testing area(now not prepared)
from sklearn.model_selection import train_test_split #train_test_split is used to separte X and Y into train and test.
from sklearn.preprocessing import StandardScaler#StandardScaler is used to standardize the Values in X of different rows.
from sklearn.linear_model import LogisticRegression#It is the algorithm used in this program for the decision makeing...
from sklearn.metrics import confusion_matrix, accuracy_score #confusion matrix to compare the data and predicted value and using accuarcy test we can predict the accuracy of our model.

dataset= pd.read_csv('TheDataSheet.csv') #reading the csv file using pandas and storing it in dataset.
sc=StandardScaler()#assigning StandardScaler as sc
model= LogisticRegression()#assiging LogisticRegression as model.
#print(dataset) #This prints the datas of the csv file(stored in dataset variable)
#print(dataset.head(5))  #used to print the first 5 rows of data.
#print(dataset.tail(5))  #used to print the last 5 rows of data.
print(dataset.shape)     #shows the number of rows and columns of data.
X=dataset.iloc[:,:-1].values #removing the last column and storing the other column is X using iloc
Y=dataset.iloc[:,-1].values  #removing all column except last column and storing it in Y using iloc
#print(X)
#print(Y)
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25, random_state=0)
#print(X_train.shape) #In the above line train_test_split is used to separate the data in X and Y into train and test.
#print(Y_train.shape) #25% of data is stored in test and other 75% data is stored in train.
#print(X_test.shape)  #random_state is used to separate the data randomly
#print(Y_test.shape)  #These separated datas are stored in train_X,train_Y,test_X,test_Y.
X_train= sc.fit_transform(X_train)  #Using StandardScaler standardizing the values in X_train using fit_transform method
X_test= sc.transform(X_test)  #Using StandardScaler standardizing the values in X_test using transform method
#print(X_test)
#print(X_train)
model.fit(X_train, Y_train) #applying algorithm(LogisticRegression) to X_train and Y_train

print("Enter age")
age=input()
print("Enter salary")
sal=input()
newcust=[[age,sal]] #storing the entered age and salary
result= model.predict(sc.transform(newcust)) # using the algorithm and datas predicting customer will buy or not.
#print(result) #result will be 1 or 0. 1 when customer buys and 0 when not sale.
if result==1:
     print("The customer will buy the Product")

else: 
    print("The customer will not by the Product")

Y_pred=model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1))
cm= confusion_matrix(Y_test, Y_pred)
print("confusion matrix: ")
print(cm)
print("Accuracy of the model: {0}%".format(accuracy_score(Y_test, Y_pred)*100))
