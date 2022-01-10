import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from joblib import dump

file_path = 'CID_features_and_results.csv'
data = pd.read_csv(file_path)

X = data[['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13','feat14','feat15','feat16','feat17','feat18','feat19','feat20','feat21','feat22','feat23','feat24','feat25','feat26','feat27','feat28','feat29','feat30','feat31','feat32','feat33','feat34','feat35','feat36','feat37','feat38']].values
Y = data[['realignedMOS']].values

Y = Y.ravel()
X = MinMaxScaler().fit_transform(X)

# separate train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y)

regr = SVR(kernel='rbf', C=300, gamma=1, verbose=1)
fitted_regr = regr.fit(X_train, y_train)
print("Train regression accuracy:", fitted_regr.score(X_train, y_train))
print("Test regression accuracy:", fitted_regr.score(X_test, y_test))
s = dump(fitted_regr, 'model.joblib')
