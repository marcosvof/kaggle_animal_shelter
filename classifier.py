import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

###### LENDO O TREINO E TESTE ######
train = pd.read_csv('input/train.csv', index_col='AnimalID')
test = pd.read_csv('input/test.csv', index_col='ID')

###### RETIRANDO O OUTCOMESUBTYPE O NAME DO TREINO E TESTE ######
train.drop('OutcomeSubtype', axis=1, inplace=True)
train.drop(['Name', 'DateTime'], axis=1, inplace=True)
test.drop(['Name', 'DateTime'], axis=1, inplace=True)

###### TRATAMENTO DOS DADOS CATEGORICOS ######
for col in ['AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']:
    le = LabelEncoder().fit(np.append(train[col], test[col]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

#print train.head(3)

###### SEPARA O TREINO EM DADOS DE TREINO E LISTA DE CLASSES ######
data   = train.drop('OutcomeType', axis=1)
target = train['OutcomeType']

###### ESCOLHA O SEU CLASSIFICADOR ######
#clf = KNeighborsClassifier(3)
#clf = SVC(probability=True)
#clf = SVC(gamma=2, C=1)
#clf = GaussianNB()
#clf = QuadraticDiscriminantAnalysis()
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = AdaBoostClassifier()
#clf = LinearDiscriminantAnalysis()
clf = GradientBoostingClassifier()

###### TREINANDO O CLASSIFICADOR ######
fit = clf.fit(data, target)

###### EXECUTANDO VALIDACAO CRUZADA E IMPRIMINDO NO TERMINAL O VALOR DA METRICA LOG_LOSS ######
print "LogLoss:"
print cross_val_score(clf, data, target, cv=3, scoring='log_loss', verbose=0)

###### DESCOMENTE AS LINHAS ABAIXO PARA GERAR UMA PREDICAO PARA O KAGGLE ######
#proba = fit.predict_proba(test)
#ret = pd.DataFrame(proba, index=test.index, columns=fit.classes_)
#ret.sort_index(inplace=True)
#ret.to_csv('output/submission.csv', index_label="ID")