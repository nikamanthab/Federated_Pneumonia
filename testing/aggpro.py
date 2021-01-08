from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/getmodel')
def getAggregatedModel():
    return 1

@app.route('/postmodel')
def sendmodel(model):
    return 1