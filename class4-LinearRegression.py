import numpy as np
import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("medical.csv")


catagorical_col = dataset.select_dtypes(exclude=["number"]).columns

#dataset = dataset.drop(columns=["gender"])

x = dataset[["age", "bmi", "children"]]        #use to predict the target value
y = dataset.iloc[:, -1]                        #targeted value

x = x.fillna(x.mean())

#test_size tells how the data will slip like 0.2 mean 20% test and 80% train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


linear = LinearRegression()
linear.fit(x_train, y_train)

print("Slope: ", linear.coef_)
print("Inercept: ", linear.intercept_)


y_pred = linear.predict(x_test)

# Regression Metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# Visualization (Age vs Charges)
plt.scatter(x_train["age"], y_train, color="red", label="Actual")
plt.scatter(x_train["age"], linear.predict(x_train), color="blue", label="Predicted")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.title("Age vs Medical Charges")
plt.legend()
plt.show()
age = float(input("Enter age: "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))

pred_data = [[age, bmi, children]]
predicted_charges = linear.predict(pred_data)

print("Predicted charges will be: ", predicted_charges[0])