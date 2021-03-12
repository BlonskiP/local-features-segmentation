import networkx as nx
import random
def random_graph(num_nodes=5,prob_edge=0.3,seed=42):
    '''

    :param num_nodes: int - number of nodes
    :param prob_edge: float - probablity of edge acc
    :param seed: int - seed for random number generator
    :return: networkx Graph - for test purpose
    '''
    random.seed(seed)
    G = nx.fast_gnp_random_graph(num_nodes, prob_edge, seed=seed, directed=False)
    for idx, node in enumerate(G._node):
        G._node[idx] = {'x': random.randint(0,100),
                        'y': random.randint(0,100)}
    return G