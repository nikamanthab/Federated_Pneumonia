from flask import Flask, request, send_from_directory, send_file
import requests
import json
import torch
import os
from test import test
from dataloader import getTestLoader
from aggregatorloader import selectAggregator
import log
from checks import checkModelEqual, compare_models
import time
import threading
from smpc import layer_sharing

from modelloader import createInitialModel
from config import Arguments
serverargs = Arguments()
serverargs['current_agg_epoch'] = 0

app = Flask(__name__)
app.debug=True

node_details = []


uploads = serverargs['aggregated_model_location']
filename = 'agg_model.pt'
if os.path.exists(os.path.join(uploads, filename)) == False:
    createInitialModel(serverargs)
    print("Model initialized")


# initialize wandb

if(serverargs['wandb']==True):
    logger = log.initialize_wandb()
else:
    logger = None

@app.route('/getConnection', methods=['GET', 'POST'])
def getConnection():
    print("Connecting Nodes:")    
    if len(node_details) < serverargs['num_of_nodes']:
        data = request.get_json()
        data['agg_epoch'] = serverargs['current_agg_epoch']
        node_details.append(data)
        print(data['node_name'])
        return json.dumps(serverargs)
    else:
        # test later
        print("Maximum limit reached!")
        return json.dumps({"status":"max_reached"})
    


@app.route('/sendmodel', methods=['GET', 'POST'])
def sendmodel():
    file = request.files['file']
    path = os.path.join(serverargs['aggregated_model_location'], \
        request.files['file'].filename)
    file.save(path)

    count_done = 0
    # update train phase of the node
    for node in node_details:
        if node['node_name'] == request.files['file'].filename.split('.')[0]:
            node['agg_epoch']+=1
        if ((node['agg_epoch']-1) == serverargs['current_agg_epoch']):
            count_done+=1
    result = {}
    if count_done == serverargs['num_of_nodes']:
        result = {"status": "doaggregation"}
    else:
        result = {"status": "model sent successfully!"}
    return json.dumps(result)

def aggregation_thread():
    # Getting test loader
    test_loader = getTestLoader(serverargs)
    # Call aggregator
    agg_func = selectAggregator(serverargs)
    model_data = []
    node_model = 0
    for node in node_details:
        node_model = torch.load(serverargs['aggregated_model_location']+node['node_name']+'.pt') \
            .to(serverargs['device'])
        # test(serverargs, node_model, test_loader, logger=logger)
        node_tuple = (node_model, node['no_of_samples'])
        model_data.append(node_tuple)
    if serverargs['smpc']:
        model_data = layer_sharing(model_data, serverargs)
    agg_model = agg_func(model_data, serverargs)
    print("ModelCheck: ", checkModelEqual(node_model, agg_model))
    compare_models(node_model, agg_model)
#         torch.save(node_model, serverargs['aggregated_model_location']+'agg_model.pt')
    torch.save(agg_model, serverargs['aggregated_model_location']+'agg_model.pt')
#         agg_model = torch.load(serverargs['aggregated_model_location']+'agg_model.pt')
#         import pdb; pdb.set_trace()
    print("---Aggregation Done---")
    #testing agg_model
#         import pdb; pdb.set_trace()
    # test(serverargs, node_model, test_loader, logger=logger)
    test(serverargs, agg_model, test_loader, logger=logger)
    serverargs['current_agg_epoch']+=1

    serverargs['aggregator'] = 'comed'
    agg_model = agg_func(model_data, serverargs)
    print("---Aggregation Done---")
    test(serverargs, agg_model, test_loader, logger=logger)


@app.route('/doaggregation', methods=['GET','POST'])
def doaggregation():
    x = threading.Thread(target=aggregation_thread, args=())
    x.start()
    return json.dumps({"status": "model sent successfully!"})

@app.route('/getmodel', methods=['GET', 'POST'])
def getModel():
    uploads = serverargs['aggregated_model_location']
    filename = 'agg_model.pt'
#     if os.path.exists(os.path.join(uploads, filename)) == False:
#         try:
#             createInitialModel(serverargs)
#             print("model initialized")
#         except:
#             print("Model Initialization error")
    return send_from_directory(uploads, filename)

@app.route('/checkphase', methods=['GET', 'POST'])
def checkPhase():
    if request.get_json()['local_agg_epoch'] == serverargs['current_agg_epoch']:
        return json.dumps({"phase": "done"})
    else:
        return json.dumps({"phase": "aggregating"})

app.run()