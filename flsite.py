import pickle

import numpy as np
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

loaded_model_knn = pickle.load(open('model/Sport_pickle_file', 'rb'))
loaded_model_lab2 = pickle.load(open('model/Boots_pickle_file', 'rb'))
loaded_model_lab3 = pickle.load(open('model/Credit_pickle_file', 'rb'))

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Ковалевым Егором", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           ]])
        pred = loaded_model_knn.predict(X_new)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model=pred)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Линейная регрессия для рассчета размера обуви", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           ]])
        pred = loaded_model_lab2.predict(X_new)
        return render_template('lab2.html', title="Линейная регрессия для рассчета размера обуви", menu=menu,
                               class_model=pred)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия одобрения кредита", menu=menu,
                               class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           ]])
        pred = loaded_model_lab3.predict(X_new)
        return render_template('lab3.html', title="Логистическая регрессия одобрения кредита", menu=menu,
                               class_model=pred)

if __name__ == "__main__":
    app.run(debug=True)
