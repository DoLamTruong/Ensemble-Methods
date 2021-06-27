from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
N = 2000
moons = make_circles(n_samples= N , noise= 0.3)
X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)

det = DecisionTreeClassifier(max_depth=1)
ada_clf = AdaBoostClassifier(
 DecisionTreeClassifier(max_depth=1), n_estimators=1000,
 algorithm="SAMME.R", learning_rate=0.5
 )
ada_clf.fit(X_train, y_train)

for clf in (det, ada_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))