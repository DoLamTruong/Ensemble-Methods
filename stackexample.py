import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
N = 1000
moons = make_moons(n_samples= N , noise= 0.3)
x_train, x_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)

# We first define a function to make predictions on n-folds of train and test dataset. This function returns the predictions for train and test for each model.
def Stacking(model,train,y,test,n_fold):
    folds=StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred = np.empty((test.shape[0],1),float)
    train_pred = np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val = train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val = y.iloc[train_indices],y.iloc[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred = np.append(train_pred,model.predict(x_val))
        test_pred = np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred

# Now we’ll create two base models – decision tree and knn.
model_1 = DecisionTreeClassifier(random_state=1)
test_pred_1 ,train_pred_1 = Stacking(model=model_1, n_fold=10, train=x_train, test=x_test, y=y_train)
train_pred_1 = pd.DataFrame(train_pred_1)
test_pred_1 = pd.DataFrame(test_pred_1)

model_2 = KNeighborsClassifier()
test_pred_2, train_pred_2 = Stacking(model=model_2, n_fold=10, train=x_train,test=x_test, y=y_train)
train_pred_2 = pd.DataFrame(train_pred_2)
test_pred_2 = pd.DataFrame(test_pred_2)

# Create a final model, logistic regression, on the predictions of the decision tree and knn models.
df = pd.concat([train_pred_1, train_pred_2], axis=1)
df_test = pd.concat([test_pred_1, test_pred_2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)