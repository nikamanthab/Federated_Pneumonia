from flask import Flask, request, send_from_directory, send_file
import requests
import json
import os
from aggregatorloader import selectAggregator

from modelloader import createInitialModel
from config import Arguments
serverargs = Arguments()

app = Flask(__name__)
app.debug=True

node_details = []


@app.route('/getConnection', methods=['GET', 'POST'])
def getConnection():
    print("Connecting Nodes:")
    if len(node_details) < serverargs['num_of_nodes']:
        data = request.get_json()
        data['phase'] = 'training'
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
    import pdb; pdb.set_trace()
    path = os.path.join(serverargs['aggregated_model_location'], \
        request.files['file'].filename)
    file.save(path)

    count_done = 0
    # update train phase of the node
    for node in node_details:
        if node['node_name'] == request.files['file'].filename.split('.')[0]:
            node['phase'] = 'done'
        if node['phase'] == 'done':
            count_done+=1
    # Call aggregator
    if count_done == len(node_details):
        agg_func = selectAggregator(serverargs)
        for node in node_details:
            node_model = torch.load(node['aggregated_model_location']).to(serverargs['device'])
            node_tuple = (node_model, node['no_of_samples'])
        agg_model = agg_func(node_tuple)
        torch.save(agg_model, serverargs['aggregated_model_location']+'agg_model.pt')
    return json.dumps({"status": "model sent successfully!"})

@app.route('/getmodel', methods=['GET', 'POST'])
def getAggregatedModel():
    # busy loop to check training status
    uploads = serverargs['aggregated_model_location']
    if os.path.exists(uploads):
        try:
            createInitialModel(serverargs)
            print("model initialized")
        except:
            print("Model Initialization error")
    filename = 'agg_model.pt'
    return send_from_directory(uploads, filename)

app.run()