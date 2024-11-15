import pickle

import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

loaded_model_knn = pickle.load(open('model/Sport_pickle_file', 'rb'))

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('speed')),
                       float(request.args.get('lengthJump')),
                       float(request.args.get('lengthHand'))]])
    pred = loaded_model_knn.predict(X_new)

    return jsonify(sort=pred[0])