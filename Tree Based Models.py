import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error as MSE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.preprocessing import scale

#DecisionTreeClassifier

data = pd.read_csv('data_breast.csv', sep  = ',')  #Cargado de la base
data = data.drop(data.iloc[:,12:], axis = 1) #Se eliminan las columnas que no sirven con _std y _worst
data = data.drop(columns = 'id') #Se elimina la columna de 'id' que no entrega información necesaria

print(data.head())
print(data.info())

print(pd.unique(data['diagnosis']))
data['diagnosis'] = pd.get_dummies(data['diagnosis'])['M']

X = data.drop(columns = 'diagnosis')
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state= 420)


dt = DecisionTreeClassifier(max_depth = 2, random_state = 420)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))


dt1 = DecisionTreeClassifier(criterion = 'gini', random_state = 420)
dt1.fit(X_train, y_train)

y_pred1 = dt1.predict(X_test)
print(accuracy_score(y_test, y_pred1))

#Hyperparameter Tuning for Trees

dt = DecisionTreeClassifier(random_state = 420)

params_dt = {'max_depth' : [3, 4, 5, 6],
             'min_samples_leaf' : [0.04, 0.06, 0.08],
             'max_features' : [0.2, 0.4, 0.6, 0.8]}



grid_dt = GridSearchCV(estimator = dt, param_grid = params_dt, 
                       scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_dt.fit(X_train, y_train)

best_hyperparams = grid_dt.best_params_
print(f'Best Hyperparameters:\n {best_hyperparams}')

best_CV_score = grid_dt.best_score_
print(f'Best CV accuracy: \n {best_CV_score}')

best_model = grid_dt.best_estimator_

test_acc = best_model.score(X_test, y_test)
print(f'Test set accuracy of BEST MODEL: {test_acc}')

#DecisionTreeRegressor

df = pd.read_csv('auto-mpg.csv')
df = df.drop(columns = 'car name')

df['horsepower'] = df['horsepower'].astype(str)
df = df[df['horsepower'] != '?']
df['horsepower'] = pd.to_numeric(df['horsepower'])

X = df.drop(columns = 'mpg')
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 3)

dt = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 0.1, random_state = 3)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

mse_dt = MSE(y_test,y_pred)
rmse_dt = mse_dt**(1/2)

#K-Fold CrossValidation

dt = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 0.14)
MSE_CV = - cross_val_score(dt, X_train, y_train, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = -1)
dt.fit(X_train,y_train)
y_predict_train = dt.predict(X_train)
y_predict_test = dt.predict(X_test)
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
print('Train MSE: {:.2f}'.format(MSE(y_train,y_predict_train)))
print('Test MSE: {:.2f}'.format(MSE(y_test,y_predict_test)))


#Ensemble Learning: Hard Voting

SEED = 1

X = data.drop(columns = 'diagnosis')
y = data['diagnosis']

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= SEED)

lr = LogisticRegression(random_state = SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state = SEED)

classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbors', knn),
               ('Classification Tree', dt)]

for clf_name, clf in classifiers:
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))


vc = VotingClassifier(estimators = classifiers)

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)

print(f'Voting Classifier: {accuracy_score(y_test, y_pred)}')

#Ensemble Learning: Bagging

SEED_1 = 2

X = data.drop(columns = 'diagnosis')
y = data['diagnosis']

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= SEED_1)

dt = DecisionTreeClassifier(max_depth = 4, min_samples_leaf=0.16, random_state = SEED_1)

bc = BaggingClassifier(base_estimator = dt, n_estimators = 500, n_jobs = -1)

bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy of Bagging Classifier: {accuracy}')

bc = BaggingClassifier(base_estimator = dt, n_estimators = 300, oob_score = True, n_jobs = -1)

bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f'OOB Accuracy: {bc.oob_score_}')

#RandomForest: Regressor 
X = df.drop(columns = 'mpg')
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 3)

rf = RandomForestRegressor(n_estimators = 400, min_samples_leaf = 0.12, random_state = 3)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rmse_test = MSE(y_test,y_pred)**(1/2)

print(f'Test set RMSE of RandomForest: {rmse_test}')

importances_rf = pd.Series(rf.feature_importances_, index = X.columns)

sorted_importances_rf = importances_rf.sort_values()

sorted_importances_rf.plot(kind = 'barh', color = 'lightgreen')

#Hyperparameter Tuning for Random Forest

params_rf= {'n_estimators' : [300, 400, 500],
          'max_depth' : [4, 6, 8],
          'min_samples_leaf' : [0.1, 0.2],
          'max_features' : ['log2', 'sqrt']}

grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 3, scoring = 'neg_mean_squared_error',
                       verbose = 1, n_jobs = -1)


grid_rf.fit(X_train, y_train)

best_hyperparams = grid_rf.best_params_
print(f'Best Hyperparameters for Random Forest:\n {best_hyperparams}')

best_model = grid_rf.best_estimator_

y_pred = best_model.predict(X_test)

rmse_test = MSE(y_test, y_pred)**(1/2)
print(f'Test set RMSE of RF: {rmse_test}')


