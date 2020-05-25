import pandas as pd
dataset = pd.read_csv('SalaryData.csv')
dataset.head()
y = dataset['Salary']
X = dataset['YearsExperience'].values.reshape(30,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
