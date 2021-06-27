from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
N = 1000
moons = make_circles(n_samples= N , noise= 0.3)
X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)
X_train2, X_t, y_train2, y_t = train_test_split(X_test, y_test, test_size=0.3, random_state=101)

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

blender = LogisticRegression()
a1= log_clf.predict(X_train2)
a2 = rnd_clf.predict(X_train2)
a3 =  svm_clf.predict(X_train2)
blenderTrain = []
for i in range(len(a1)):
    blenderTrain += [[a1[i], a2[i], a3[i]]]
blender.fit(blenderTrain, y_train2)
# ============================================
b1= log_clf.predict(X_t)
b2 = rnd_clf.predict(X_t)
b3 =  svm_clf.predict(X_t)
blenderTest = []
for i in range(len(b1)):
    blenderTest += [[b1[i], b2[i], b3[i]]]
y_pred = blender.predict(blenderTest)

print('blender', accuracy_score(y_t, y_pred))

for clf in (log_clf, rnd_clf, svm_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_t)
    print(clf.__class__.__name__, accuracy_score(y_t, y_pred))
