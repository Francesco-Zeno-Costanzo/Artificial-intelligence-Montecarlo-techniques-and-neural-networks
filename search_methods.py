"""
Code that implement sevarl alghoritms for searching
"""
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

random.seed(69420)

def load_graph(file_path):
    '''
    Function that read a file and create the associeted graph.
    Each key in the dictionary is a node and the associated value
    is another dictionary containing all the nodes linked to the
    key node with its related infoion value.

    Parameters
    ----------
    file_path : str
        path of the file with the data
    
    Returns
    -------
    graph : dict
        graph in the form of a dictionary
    '''
    graph = {} # Dictionary that will contain the graph

    with open(file_path, 'r') as f:
        for l in f: # Read the file
            city_a, city_b, p_cost = l.split(",")
            # Remove case sensitive
            city_a = city_a.lower()
            city_b = city_b.lower()
            
            # Create a node if we read a new element
            if city_a not in graph:
                graph[city_a] = {}
            if city_b not in graph:
                graph[city_b] = {}

            # Create the "direct" link
            graph[city_a][city_b] = int(p_cost)            
            # Create the "inverse" link
            graph[city_b][city_a] = int(p_cost)
    
    return graph


def generate_random_graph(num_nodes, num_edges):
    '''
    Generate a random weighted graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of edges (connections) between nodes.

    Returns
    -------
    graph : dict
        A dictionary representing the graph.
    '''

    min_weight = 1
    max_weight = 10
    nodes      = [f"node_{i}" for i in range(num_nodes)]
    graph      = {node: {} for node in nodes}
    edges      = set()

    while len(edges) < num_edges:
        # Choose randomly two nodes
        a, b = random.sample(nodes, 2)

        if (a, b) not in edges and (b, a) not in edges:
            # If not exisit create the link
            weight = random.randint(min_weight, max_weight)
            graph[a][b] = weight
            graph[b][a] = weight
            edges.add((a, b))

    return graph


def Search(graph, src, dst, method='Breadth First', level=None):
    '''
    Function that implements search methods.
    The aviable metods are: 
    1) Breadth First
    2) Depth First
    3) Limited Depth

    Parameters
    ----------
    graph : dict
        dictionary created by load_graph function
    src : str
        name of the source node
    dst : str
        name of the destination node
    method : str, optional, default 'Breadth First'
        method used for search, if Limited Depth is
        used a value for level must also be passed.
    level : None
        level of maximum depth, mandatory if the
        selecte method is 'Limited Depth'
    
    Returns
    -------
    path : list
        list of al node form src to dst
    '''
    # For case sensitive
    src = src.lower()
    dst = dst.lower()

    # Usefull variables to switch methods
    idx   = 0 if method == 'Breadth First' else -1
    level = np.inf if method != "Limited Depth" else level
    if method == 'Limited Depth' and level is None:
        raise Exception('For Limited Depth method you must pass a level')
    
    # Variable used in the search
    info    = [(src, [src], 0)]
    visited = {src}

    while info: # Until i can search
        # Unpack the informations
        (node, path, cost) = info.pop(idx)

        # For all nearby nodes
        for temp in graph[node].keys():

            # If I find the destination I've finished
            if temp == dst:
                return path + [temp]
            
            else:
                # Oterwise, if temp is a new node
                if temp not in visited:
                    # Add temp to visited nodes, to avoid waste of time
                    visited.add(temp)
                    # If I don't exceed the limit I'll continue the search
                    if len(path) < level:
                        info.append((temp, path + [temp], cost + graph[node][temp]))


def iterative_deepening_search(graph, src, dst):
    '''
    Function that implements iterative deepening search method.

    Parameters
    ----------
    graph : dict
        dictionary created by load_graph function
    src : str
        name of the source node
    dst : str
        name of the destination node
    
    Returns
    -------
    path : list
        list of al node form src to dst
    '''
    
    level = 0  # Depth level
    check = 1  # Flag to control code stopping
    while check == 1:

        level += 1  # Increase the depth
        path   = Search(graph, src, dst, method='Limited Depth', level=level)

        if path is None:
            # If path is None level is to small we need to increase it
            continue # Restart the loop
        else :
            # Otherwise we have our result and we can stop
            check = 0
    return path


def plot_graph(graph):
    '''
    Plot a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    graph : dict
        Graph in dictionary format.
    '''
    G = nx.Graph()

    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)

    pos    = nx.spring_layout(G, seed=69420)  # Layout for visualization
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(1)
    nx.draw(G, pos, with_labels=True, node_size=1000,
            node_color="lightblue", edge_color="gray", font_size=10)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    plt.title("Graph Representation")
    plt.show()


if __name__ == "__main__":

    graph = load_graph('data.txt')    
    
    print(Search(graph, "Oradea", "Bucharest"))
    print(iterative_deepening_search(graph, "Oradea", "Bucharest"))
    plot_graph(graph)

    graph = generate_random_graph(16, 22)
    
    print(Search(graph, "Node_2", "Node_7"))
    print(iterative_deepening_search(graph, "Node_2", "Node_7"))
    plot_graph(graph)

