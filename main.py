import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
from sklearn.svm import SVC
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from abc import ABC, abstractmethod


# try after modeling removing the outliers, because a natural outliers can be helpful test if it is better without
# them

class predicting(ABC):

    @abstractmethod
    def fit(self, X_train=0, y_train=0):
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def saveModel(self):
        pass

    @abstractmethod
    def loadModel(self):
        pass


class Dtree(predicting):
    model = tree.DecisionTreeClassifier(random_state=2)

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)


    def score(self, X_test, y_test):
        print("Dtree predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'Dtree.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'Dtree.sav'
        self.model = pickle.load(open(filename, 'rb'))
    def plot(self):
        tree.plot_tree(self.model)


class logistic(predicting):
    model = LogisticRegression(solver='lbfgs', max_iter=10000)

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("Logistic predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'Logistic.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'Logistic.sav'
        self.model = pickle.load(open(filename, 'rb'))


class knn(predicting):
    model = KNeighborsClassifier(n_neighbors=3)

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)


    def score(self, X_test, y_test):
        print("knn predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'knn.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'knn.sav'
        self.model = pickle.load(open(filename, 'rb'))


class bayes(predicting):
    model = GaussianNB()

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("bayes predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'bayes.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'bayes.sav'
        self.model = pickle.load(open(filename, 'rb'))


class SVMrbf(predicting):
    model = SVC(kernel='rbf')

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("SVMrbf predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'SVMrbf.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'SVMrbf.sav'
        self.model = pickle.load(open(filename, 'rb'))


class SVMlinear(predicting):
    model = SVC(kernel='linear')

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("SVMlinear predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'SVMlinear.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'SVMlinear.sav'
        self.model = pickle.load(open(filename, 'rb'))


class SVMpoly(predicting):
    model = SVC(kernel='poly')

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("SVMpoly predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'SVMpoly.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'SVMpoly.sav'
        self.model = pickle.load(open(filename, 'rb'))


class SVMsigmoid(predicting):
    model = SVC(kernel='sigmoid')

    def __init__(self, X_train=0, y_train=0, mode=''):
        if mode == 'train':
            self.model.fit(X_train, y_train)

        elif mode == 'load':
            self.loadModel()

    def fit(self, X_train=0, y_train=0):
        self.model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        print("SVMsigmoid predict score is: ", self.model.score(X_test, y_test))

    def predict(self, X):
        return self.model.predict(X)

    def saveModel(self):
        filename = 'SVMsigmoid.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def loadModel(self):
        filename = 'SVMsigmoid.sav'
        self.model = pickle.load(open(filename, 'rb'))


class dataPreprocessing:
    def __init__(self):
        self.self = self

    def returning(self, directory):
        print(directory)
        df = pd.read_csv(directory)
        pd.set_option('display.max_rows', 1000)

        # data cleaning
        df = df.drop(columns='Index', axis=0)  # droping index col
        df = df.drop_duplicates()  # droping duplicated rows if existed
        df = df.dropna(how='any', axis=0)  # droping Nan rows if existed

        # testing removing the outliers
        # var = df[np.abs(df.Data - df.Data.mean()) <= (3 * df.Data.std())]
        ##end test

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        # print(df.describe())
        # print(df.duplicated()) #checking for duplicated values

        # feature selection: using feature extraction for supervised learning
        # Correlated features will not always improve the model but might overfit, so we remove high corr features
        # Too many features can lead to overfitting because the higher the complexity of the model
        cor_matrix = df.corr().abs()
        # print(cor_matrix)
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        df1 = df
        for i in to_drop:
            df1 = df1.drop(i, axis=1)
        # print(df1)

        return df1


directory = 'Test.csv'
data = dataPreprocessing()
df = data.returning(directory)

y = df['diagnosis']
X = df.drop(columns='diagnosis', axis=0)  # dropping the diagnosis column
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=564123)

#########models training and score
# models can have the same exact score when the dataset is small

#### Tree model training ####
clf = Dtree(X_train, y_train, mode='train')
clf.score(X_test, y_test)
res1 = clf.predict(X_test)
clf.plot()
# clf.saveModel()
# clf = Dtree()
# clf.loadModel()
# clf.score(X_test,y_test)

#### SVM model training Linear ####
SVM = SVMlinear(X_train, y_train, mode='train')
SVM.score(X_test, y_test)
res2 = SVM.predict(X_test)
# SVM.saveModel()
# SVM = SVMlinear()
# SVM.loadModel()
# SVM.score(X_test,y_test)

#### Logistic model training ####
logistic_model = logistic(X_train, y_train, mode='train')
logistic_model.score(X_test, y_test)
res3 = logistic_model.predict(X_test)
# logistic_model.saveModel()
# logistic_model = logistic()
# logistic_model.loadModel()
# logistic_model.score(X_test,y_test)

#### KNN model training ####
neigh = knn(X_train, y_train, mode='train')
neigh.score(X_test, y_test)
res4 = neigh.predict(X_test)
# neigh.saveModel()
# neigh = knn()
# neigh.loadModel()
# neigh.score(X_test,y_test)

#### SVM model training rbf ####
SVMBrf = SVMrbf(X_train, y_train, mode='train')
SVMBrf.score(X_test, y_test)
res5 = SVMBrf.predict(X_test)
# SVMBrf.saveModel()
# SVMBrf = SVMrbf()
# SVMBrf.loadModel()
# SVMBrf.score(X_test,y_test)

#### SVM model training poly ####
SVMPoly = SVMpoly(X_train, y_train, mode='train')
SVMPoly.score(X_test, y_test)
res6 = SVMPoly.predict(X_test)
# SVMPoly.saveModel()
# SVMPoly = SVMpoly()
# SVMPoly.loadModel()
# SVMPoly.score(X_test,y_test)

#### SVM model training sigmoid ####
SVMSigmoid = SVMsigmoid(X_train, y_train, mode='train')
SVMSigmoid.score(X_test, y_test)
res7 = SVMSigmoid.predict(X_test)
# SVMSigmoid.saveModel()
# SVMSigmoid = SVMsigmoid()
# SVMSigmoid.loadModel()
# SVMSigmoid.score(X_test,y_test)

#### bayes ####
gnb = bayes(X_train, y_train, mode='train')
gnb.score(X_test, y_test)
res8 = gnb.predict(X_test)
# gnb.saveModel()
# gnb = bayes()
# gnb.loadModel()
# gnb.score(X_test,y_test)

newDataPath = input("Enter your Directory: ")
newData = dataPreprocessing()
newDf = data.returning(directory)

arr = [""]
for i in range(len(res1)):
    b = 0
    m = 0
    if res1[i] == 'B':
        b += 1
    else:
        m += 1
    if res2[i] == 'B':
        b += 1
    else:
        m += 1
    if res3[i] == 'B':
        b += 1
    else:
        m += 1
    if res4[i] == 'B':
        b += 1
    else:
        m += 1
    if res5[i] == 'B':
        b += 1
    else:
        m += 1
    if res6[i] == 'B':
        b += 1
    else:
        m += 1
    if res7[i] == 'B':
        b += 1
    else:
        m += 1
    if res8[i] == 'B':
        b += 1
    else:
        m += 1

    if b > m:
       # print('B ', b)
        arr.append('B')
    else:
        #print('M ', m)
        arr.append('M')

arr.remove('')
y_test = y_test.to_numpy()

##test if it was true, add the data to then split it to x_test and y_test
errorNum = 0
for i in range(len(arr)):
    if arr[i] != y_test[i]:
        errorNum += 1
        #print(i , "error in ", arr[i], ' ', y_test[i])

print(errorNum, " ", 1 - (errorNum / len(arr)))


plt.show()


