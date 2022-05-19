import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("STEP1")
# Reading data
df = pd.read_excel("Python data.xlsx")
X = df[['study_abroad', 'female', 'internship', 'father_highschool', 'father_university', 'mother_highschool','mother_university']]
y = df.work_abroad

# Train/Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Choice of regression model (Logistic Regression)
reg = linear_model.LogisticRegression(solver='lbfgs')

# Model training
reg.fit(X_train, y_train)

# Model metrics on training data
print("Accuracy on training data : ", accuracy_score(y_train, reg.predict(X_train)))
print("Mean squared error on training data : ", mean_squared_error(y_train, reg.predict(X_train)))
print("R² score on training data : ", r2_score(y_train, reg.predict(X_train)))

# Model metrics on test data
print("Accuracy on test data : ", accuracy_score(y_test, reg.predict(X_test)))
print("Mean squared error on test : ", mean_squared_error(y_test, reg.predict(X_test)))
print("R² score on training data : ", r2_score(y_test, reg.predict(X_test)))

print("STEP2")

X1 = df[['study_abroad', 'female', 'internship', 'father_highschool', 'father_university', 'mother_highschool','mother_university', 'university', 'department']]

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.33, random_state=42)

reg1 = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

reg1.fit(X1_train, y_train)

# Model metrics on training data
print("Accuracy on training data : ", accuracy_score(y_train, reg1.predict(X1_train)))
print("Mean squared error on training data : ", mean_squared_error(y_train, reg1.predict(X1_train)))
print("R² score on training data : ", r2_score(y_train, reg1.predict(X1_train)))

# Model metrics on test data
print("Accuracy on test data : ", accuracy_score(y_test, reg1.predict(X1_test)))
print("Mean squared error on test : ", mean_squared_error(y_test, reg1.predict(X1_test)))
print("R² score on training data : ", r2_score(y_test, reg1.predict(X1_test)))

print("STEP3")
X2 = df[['study_abroad', 'female', 'internship', 'father_highschool', 'father_university', 'mother_highschool','mother_university', 'university', 'department', 'ERASMUS']]

X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=42)

reg2 = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

reg2.fit(X2_train, y_train)

# Model metrics on training data
print("Accuracy on training data : ", accuracy_score(y_train, reg2.predict(X2_train)))
print("Mean squared error on training data : ", mean_squared_error(y_train, reg2.predict(X2_train)))
print("R² score on training data : ", r2_score(y_train, reg2.predict(X2_train)))

# Model metrics on test data
print("Accuracy on test data : ", accuracy_score(y_test, reg2.predict(X2_test)))
print("Mean squared error on test : ", mean_squared_error(y_test, reg2.predict(X2_test)))
print("R² score on training data : ", r2_score(y_test, reg2.predict(X2_test)))

# Plots
df.groupby('university')['work_abroad'].value_counts().unstack().plot.bar()
plt.show()

df.groupby('university')['study_abroad'].value_counts().unstack().plot.bar()
plt.show()

df.groupby('university')['work_abroad', 'study_abroad'].sum().plot(kind='bar', legend='True')
plt.show()



