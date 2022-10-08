from dataframes import dataset


from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

#testing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



ds = dataset("files/train.csv")

ds.run_default(10,10)

X_train, y_train = ds.get_Train()

ds.init_test_data("files/test.csv", "PassengerId")

knn = KNeighborsClassifier(n_neighbors=83)


regression = LogisticRegression(solver='liblinear', random_state=0)
regression.fit(X_train, y_train)

X_test = ds.run_test_deafault()
pred = regression.predict(X_test)

ds.to_csv(pred,'Transported', 'files/LogisticRegression_submissions.csv')