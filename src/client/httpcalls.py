import requests
import os
import time

def getConnection(url, node_details):
    return requests.post(url+'/getConnection', json=node_details).json()
    
def getModel(url, path, local_agg_epoch):
    while(True):
        response = requests.post(url+'/checkphase', \
            json={"local_agg_epoch": local_agg_epoch} \
                )
        if response.json()['phase'] == 'aggregating':
            print("Waiting for aggregation...")
            time.sleep(100)
            continue
        else:
            break
    response = requests.post(url+'/getmodel', stream=True)
    model_file = open(path,"wb")
    for chunk in response.iter_content(chunk_size=1024):
        model_file.write(chunk)
    model_file.close()

def sendModel(url, path, args):
    res = requests.post(url+'/sendmodel', files={'file': (args['node_name']+'.pt', open(path, 'rb'))}, stream=True)
    if res.json()['status'] == 'doaggregation':
        final_res = requests.post(url+'/doaggregation')
    print(res.json()['status'])