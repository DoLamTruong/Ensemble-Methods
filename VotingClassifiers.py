from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
N = 1000
moons = make_moons(n_samples= N , noise= 0.3)
X_train, X_test, y_train, y_test = train_test_split(moons[0], moons[1], test_size=0.3, random_state=101)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
 estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
 voting='hard'
 )
# voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# class1 = [[],[]]
# class2 = [[],[]]
# for i in range(N):
#     if moons[1][i] == 1:
#         class2[0].append(moons[0][i][0])
#         class2[1].append(moons[0][i][1])
#     else:
#         class1[0].append(moons[0][i][0])
#         class1[1].append(moons[0][i][1])

# import matplotlib.pyplot as plt
# plt.plot(class1[0],class1[1] , 'ro')
# plt.plot(class2[0], class2[1], 'bs')
# plt.show()