import pandas as pd
import random
import argparse




def datasplit():
    parser = argparse.ArgumentParser(description='Server module.')
    parser.add_argument('--num_nodes', type=int, default=5)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file',type=str)
    args = parser.parse_args()

    df=pd.read_csv(args.input_file)
    num_nodes = args.num_nodes

    length=len(df)
    numrows = int(length/num_nodes)
    nodes = [[] for i in range(num_nodes) ]
    count=0
    while(count<num_nodes):
        length = len(df)
        indices = random.sample(range(0, length), numrows)
        nodes[count] = df.loc[indices]
        df = df.drop(indices)
        df=df.reset_index(drop=True)
        count+=1
    
    for i in range(num_nodes):            
        temp=pd.DataFrame(nodes[i])
        filename = "node_" + str(i) + ".csv"
        # filepath = "../csv/splits/" + filename 
        filepath = args.output_file + filename
        temp.to_csv(path_or_buf=filepath)
        

datasplit()
        
        
        