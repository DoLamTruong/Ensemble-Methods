from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
N = 1000
moons = load_diabetes(load_diabetes(return_X_y=True))

X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)

det = DecisionTreeRegressor()
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X_train, y_train)
gbrt.predict(X_test)