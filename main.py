# from flask import Flask, render_template
# app = Flask(__name__)


# @app.route('/')
# def hello_world():
#     return render_template('index.html')


# app.run()


# this is my test of Flask #

# from flask import Flask, jsonify, request
# from flask_mysqldb import MySQL
# from flask_cors import CORS

# app = Flask(__name__)

# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'crovy'
# app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# mysql = MySQL(app)

# CORS(app)


# @app.route('/api/tasks', methods=['GET'])
# def get_all_tasks():
#      cur = mysql.connection.cursor()
#      cur.execute("SELECT * FROM cholesterol where id=(SELECT MAX(id) FROM cholesterol where user_id3=2)")
#      rv = cur.fetchall()
#      return jsonify(rv)

#  #return jsonify({'text':'Hello World!'})


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, jsonify, request
from sklearn import svm
from sklearn import datasets
from flask_cors import CORS

# from sklearn.externals import joblib
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
import csv

# initialize flask application
app = Flask(__name__)
CORS(app)
##########################################################################
# Setup the Flask-JWT-Extended extension
app.config['JWT_SECRET_KEY'] = 'super-secret'  # Change this!
jwt = JWTManager(app)


# Provide a method to create access tokens. The create_access_token()
# function is used to actually generate the token, and you can return
# it to the caller however you choose.
@app.route('/login', methods=['POST'])
def login():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if not username:
        return jsonify({"msg": "Missing username parameter"}), 400
    if not password:
        return jsonify({"msg": "Missing password parameter"}), 400

    if username != 'test' or password != 'test':
        return jsonify({"msg": "Bad username or password"}), 401

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200


##########################################################################

@app.route('/api/train', methods=['GET'])
@jwt_required
def train():
    df = pd.read_csv('./mathiew1.csv')
    target = 'class'
    features = ['age', 'sex', 'blood pressure', 'cholesterol', 'maximum heart rate', 'angina']
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modell = LogisticRegression()
    predictions = modell.fit(X_train, y_train).predict(X_test)
    joblib.dump(modell, 'model.pkl')
    return jsonify({'accuracy': round(accuracy_score(y_test, predictions) * 100, 2)})


@app.route('/api/predict', methods=['POST'])
@jwt_required
def predict():
    X = request.get_json()
    X = [[X['age'], X['sex'], X['bloodPressure'], X['cholesterol'], X['maximumHeart'], X['angina']]]
    clf = joblib.load('model.pkl')
    # convert data into dataframe
    data_df = pd.DataFrame.from_dict(X)

    # predictions
    result = clf.predict(X)
    # send back to browser
    output = {'results': int(result[0])}
    # return data
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)