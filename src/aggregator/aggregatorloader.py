import aggregator

def selectAggregator(args):
    '''
        *** Include the aggregrator functions in the dict ***
        Input: Gets the arguments
        Output: aggregator function
    '''
    switcher = {
        'comed': aggregator.comed_aggregator,
        'fedavg': aggregator.fed_avg_aggregator,
        'geomed': aggregator.Geometric_Median
    }
    agg_func = switcher.get(args['aggregator'], "aggregator name mismatch - check help for aggregator")
    return agg_func
