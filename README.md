# chance-of-admission
# Step 1 : import library
import pandas as pd
    
# Step 2 : import data
admission = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Admission%20Chance.csv')
     
admission.head()
admission.info()
admission.describe()

# Step 3 : define target (y) and features (X)
admission.columns

y = admission['Chance of Admit ']
X = admission.drop(['Serial No','Chance of Admit '],axis=1)
     
# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
     
# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Step 5 : select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
     
# Step 6 : train or fit model
model.fit(X_train,y_train)

model.intercept_
model.coef_

# Step 7 : predict model
y_pred = model.predict(X_test)
     
y_pred
# Step 8 : model accuracy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
     
mean_absolute_error(y_test,y_pred)

mean_absolute_percentage_error(y_test,y_pred)

mean_squared_error(y_test,y_pred)
