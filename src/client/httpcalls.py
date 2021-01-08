import requests
import os

def getConnection(url, node_details):
    return requests.post(url+'/getConnection', json=node_details)
    
def getModel(url, path):
    response = requests.post(url+'/getmodel',stream=True)
    model_file = open(path,"wb")
    for chunk in response.iter_content(chunk_size=1024):
        model_file.write(chunk)
    model_file.close()

def sendModel(url, path, args):
    res = requests.post(url+'/sendmodel', files={'file': (args['node_name']+'.pt', open(path, 'rb'))}, stream=True)
    print(res.json()['status'])