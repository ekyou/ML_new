import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

sport_df = pd.read_excel("sport.xlsx")
X = sport_df.iloc[:, :-1]
Y = sport_df.iloc[:, -1]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.3, random_state=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)
with open('Sport_pickle_file', 'wb') as pkl:
    pickle.dump(model, pkl)
