# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import matplotlib.pyplot as mat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print("Data imported")
data.head(10)
#plot graph
data.plot(x='Hours', y='Scores',style = 'x')
mat.title('Hours vs Percentage')
mat.xlabel('The Hours Studied')
mat.ylabel('The Percentage Score')
mat.show()

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train, y_train)
print("Training done")

line=regressor.coef_*x+regressor.intercept_

mat.scatter(x,y)
mat.plot(x,line)
mat.show()

print(x_test)
y_pred =regressor.predict(x_test)

cmp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(cmp)

hours = [[9.25]]
own_pred = regressor.predict(hours)
print("Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

print("Absolute Error = {}".format(metrics.mean_absolute_error(y_test,y_pred)))
