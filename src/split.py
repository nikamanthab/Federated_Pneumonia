import pandas as pd
import random
import argparse
import os

def datasplit(num_nodes, input_path='../../csv/original_train.csv', output_dir='../../csv/splits/', mode="min", runtype=1):
    df = pd.read_csv(input_path)
    df['folder'] = df['label']
    df = df.sample(frac=1)
    pneumonia = df[df['label'] == 'PNEUMONIA']
    normal = df[df['label'] == 'NORMAL']
    # pneumonia = df[df['label'] == 'cat']
    # normal = df[df['label'] == 'dog']
    
    max_count = max(df['label'].value_counts()[1], df['label'].value_counts()[0])
    min_count = min(df['label'].value_counts()[1], df['label'].value_counts()[0])
    
    each_len = min_count//num_nodes
    
    if(runtype ==1):
        for i in range(num_nodes):
            node_df = pd.DataFrame()
            node_df = pd.concat([node_df, pneumonia[i*each_len: (i+1)*each_len]])
            node_df = pd.concat([node_df, normal[i*each_len: (i+1)*each_len]])
            node_df = node_df.sample(frac=1)
            node_df.to_csv(os.path.join(output_dir, 'node_'+str(i)+'_train.csv'), index=False)
        
    elif(runtype == 2):
        indices = random.sample(range(0, num_nodes), int(num_nodes*0.4))
        print("Corrupt nodes :", indices)
        for i in range(num_nodes):
            node_df = pd.DataFrame()
            node_df = pd.concat([node_df, pneumonia[i*each_len: (i+1)*each_len]])
            node_df = pd.concat([node_df, normal[i*each_len: (i+1)*each_len]])
            if(i in indices):
                node_df["temp"]= np.random.randint(0, 2, node_df.shape[0])
                node_df.loc[node_df['temp'] == 0,'label'] = 'NORMAL'
                node_df.loc[node_df['temp'] == 1, 'label'] = 'PNEUMONIA'
                node_df = node_df.drop(['temp'], axis=1)
            node_df = node_df.sample(frac=1)
            node_df.to_csv(os.path.join(output_dir, 'node_'+str(i)+'_train.csv'), index=False)
    
    elif(runtype == 3):
        indices = random.sample(range(0, num_nodes), int(num_nodes*0.4))
        print("Corrupt nodes :",indices)
        for i in range(num_nodes):
            node_df = pd.DataFrame()
            node_df = pd.concat([node_df, pneumonia[i*each_len: (i+1)*each_len]])
            node_df = pd.concat([node_df, normal[i*each_len: (i+1)*each_len]])
            if(i in indices):
                node_df['label']='NORMAL'
            node_df = node_df.sample(frac=1)
            node_df.to_csv(os.path.join(output_dir, 'node_'+str(i)+'_train.csv'), index=False)
        
parser = argparse.ArgumentParser(description='Server module.')
parser.add_argument('--num_nodes', type=int, default=5)
parser.add_argument('--input_csv', type=str, default='../csv/original_train.csv')
parser.add_argument('--output_dir',type=str, default='../csv/splits/')
parser.add_argument('--mode', type=str, default='min')
parser.add_argument('--runtype',type=int, default='1')
args = parser.parse_args()


datasplit(args.num_nodes, args.input_csv, args.output_dir, args.mode, args.runtype)

# def datasplit():
#     parser = argparse.ArgumentParser(description='Server module.')
#     parser.add_argument('--num_nodes', type=int, default=5)
#     parser.add_argument('--input_file', type=str)
#     parser.add_argument('--output_file',type=str)
#     args = parser.parse_args()

#     df=pd.read_csv(args.input_file)
#     num_nodes = args.num_nodes

#     length=len(df)
#     numrows = int(length/num_nodes)
#     nodes = [[] for i in range(num_nodes) ]
#     count=0
#     while(count<num_nodes):
#         length = len(df)
#         indices = random.sample(range(0, length), numrows)
#         nodes[count] = df.loc[indices]
#         df = df.drop(indices)
#         df=df.reset_index(drop=True)
#         count+=1
    
#     for i in range(num_nodes):            
#         temp=pd.DataFrame(nodes[i])
#         filename = "node_" + str(i) + ".csv"
#         # filepath = "../csv/splits/" + filename 
#         filepath = args.output_file + filename
#         temp.to_csv(path_or_buf=filepath)

# datasplit()
        
        
        