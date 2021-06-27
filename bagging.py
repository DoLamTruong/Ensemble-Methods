from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
N = 1000
moons = make_moons(n_samples= N , noise= 0.3)
X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)

# this is an example of bagging,
# but if you want to use pasting instead, just set bootstrap=False)
det = DecisionTreeClassifier()
bag_clf = BaggingClassifier(
 DecisionTreeClassifier(), n_estimators=500,
 max_samples=100, bootstrap=True, n_jobs=-1
 )
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)

for clf in (det, bag_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1,max_samples = 100)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print('Random Forest Classifier', accuracy_score(y_test, y_pred_rf))