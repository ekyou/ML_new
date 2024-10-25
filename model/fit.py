import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

labelencoder = LabelEncoder()

sport_df = pd.read_excel("sport.xlsx")
X1 = sport_df.iloc[:, :-1]
Y1 = sport_df.iloc[:, -1]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.3, random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('Sport_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)

boots_df = pd.read_excel("HugeBoots.xlsx")
boots_df.iloc[:, 2] = labelencoder.fit_transform(boots_df.iloc[:, 2])
X2 = boots_df.iloc[:, :-1]
Y2 = boots_df.iloc[:, -1]
X_train2, X_test2, Y_train2, Y_text2 = train_test_split(X2, Y2, test_size=0.3, random_state=3)
model = LinearRegression()
model.fit(X_train2.values, Y_train2)
with open('Boots_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)

ct_df=pd.read_excel("credit.xlsx")
ct_df.iloc[:, 4]=labelencoder.fit_transform(ct_df.iloc[:, 4])
X3 = ct_df.iloc[:, :-1]
Y3 = ct_df.iloc[:, -1]
X_train3,X_test3,Y_train3,Y_test3 = train_test_split(X3,Y3,test_size=0.3,random_state=3)
model = LogisticRegression(solver='liblinear', C=1.0)
model.fit(X_train3.values,Y_train3)
with open('Credit_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)