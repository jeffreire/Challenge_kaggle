import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#
def _filling_numeric_columns(dt, colum):
    mean = dt[colum].mean()
    mean = math.floor(mean)
    dt.update(dt[colum].fillna(mean))

# obtendo os dados do arquivo
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

# removendo as colunas inrrelevantes
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace= True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace= True)

# preenchendo as linhas vazias com a media
_filling_numeric_columns(train, ['Age'])
_filling_numeric_columns(test, ['Age'])
_filling_numeric_columns(test, ['Fare'])

# preechendo as linhas vazias das colunas categoricas
train['Embarked'].fillna('no', inplace= True)

# separando os dados para realizar a previsao
X = train.iloc[:, [0,2,3,4,5,6,7,8] ].values
y = train.iloc[:, 1].values

# separando os dados do arquivo teste
X_t = test.iloc[:, [0,1,2,3,4,5,6,7]].values

# convertendo os dados categoricos do arquivo traine
oneHotEncoder = OneHotEncoder(categories='auto')
scalar_features = X[:, :2]
scalar_features1 = X[:, 3:7]
states = oneHotEncoder.fit_transform( X[:, 2].reshape(-1,1) ).toarray()
states1 = oneHotEncoder.fit_transform( X[:,7].reshape(-1,1) ).toarray()

# confirmando e deletando a ultima coluna
states = np.delete( states, np.s_[0], axis=1)
states1 = np.delete( states1, np.s_[0], axis=1)

# Concatenando as novas colunas 
X = np.concatenate( (scalar_features, states, scalar_features1, states1), axis=1)

# convertendo os dados categoricos do arquivo test
oneHotEncoder = OneHotEncoder(categories='auto')
scalar_features_t = X_t[:, :2]
scalar_features1_t = X_t[:, 3:7]
states_t = oneHotEncoder.fit_transform( X_t[:, 2].reshape(-1,1) ).toarray()
states1_t = oneHotEncoder.fit_transform( X_t[:, 7].reshape(-1,1) ).toarray()

# confirmando e deletando a ultima coluna
states_t = np.delete( states_t, np.s_[0], axis=1)
states1_t = np.delete( states1_t, np.s_[0], axis=1)
X_t = np.concatenate( (scalar_features_t, states_t, scalar_features1_t, states1_t), axis=1)

# removendo a coluna ids e separando os dados para o nosso modelo
x = X[:, [1,2,3,4,5,6,7]]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 0)

# aplicando os dados ao modelo de classificacao
RandomForest = GradientBoostingClassifier(random_state=0, n_estimators= 110)
RandomForest.fit(X_train, y_train)
# prevendo o resultado
y_pred = RandomForest.predict(X_test)
print(round(accuracy_score(y_pred, y_test) * 100, 2))

# separando os dados de test para a previsao final
y_t = X_t[:, [1,2,3,4,5,6,7]]

# separando os ids
ids = test['PassengerId']
predictions = RandomForest.predict(y_t)
# criando um novo arquivo csv e exportando os resultados 
submission = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions.astype('int64') })
submission.to_csv('submission.csv', index=False)