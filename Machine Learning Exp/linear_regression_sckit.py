import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error


#create and graph data
np.random.seed(42)

num_samples = 500

years_of_experience = np.random.randint(2, 21, size=num_samples)

slope = (200000 - 60000) / 18
intercept = 60000

salary = slope * years_of_experience + intercept + np.random.normal(0, 10_000, num_samples )

data = {'Years_of_Experience': years_of_experience, 'Salary': salary}
df = pd.DataFrame(data)


plt.figure(figsize = (10,6))
sns.scatterplot(x ='Years_of_Experience', y = 'Salary', data = df, color = 'blue')
sns.regplot(x ='Years_of_Experience', y = 'Salary', data = df, scatter = False, color = 'red')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary")

plt.show()




#Split data in train and test to detect overfitting
#overfitting: when am model emorizes noisy training examples instead of actually learning them 
X = df[["Years_of_Experience"]]
y = df["Salary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.2, random_state = 42) #assigns test_size to test_size*100 


lr = LinearRegression() #Returns R^2, the closer to one the better



lr.fit(X_train, y_train)  #learn salary formula and it's parameters, and test is held back to see how well the formula works on unseen people

train_r2 = lr.score(X_train, y_train)
print("Train R^2:", train_r2) #returns R^2, closer to 1, the better (calculates how well the already-trained model fits the training data.(X_train, y_train))

test_r2 = lr.score(X_test, y_test)
print("Test R^2:", test_r2 ) #close train and test R^2 imply that there was no overfitting


y_test_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
#difference between predicted values and actual values
mae = mean_absolute_error(y_test, y_test_pred)
print("Mae:",mae)

train_mse = mean_squared_error(y_train, y_train_pred )
test_mse = mean_squared_error(y_test,y_test_pred)

print("Train_mse:", train_mse)
print("Test_mse:", test_mse)


test_r2 = r2_score(y_test, y_test_pred)
print("Test R^2", test_r2)

lr_coef = lr.coef_
print("Linear Coefficient: ", lr_coef[0])


lr_intercept = lr.intercept_
print("Linear intercept:", lr_intercept)


X = np.linspace(0, 20, 100)
y = lr_coef*X + intercept

plt.scatter(X, y, label=f'y = {lr_coef[0]}x + {intercept}')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Salary")

plt.grid()
plt.show()
