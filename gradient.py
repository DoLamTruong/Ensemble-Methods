from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from numpy import random

N = 200
initX, initY = load_diabetes(return_X_y=True)

X, X_test, y, y_test = train_test_split(initX, initY, test_size=0.3, random_state=101)

tree_reg1 = DecisionTreeRegressor(max_depth=1)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=1)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=1)
tree_reg3.fit(X, y3)
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(y_pred)
print(y_test)
