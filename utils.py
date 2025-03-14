"""
Functions for loading, creating and plotting graphs and grids
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
    key node with its related info on value.

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
            # Remove case-sensitive
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


def generate_random_graph(num_nodes, num_edges, min_w=0, max_w=1):
    '''
    Generate a random weighted graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of edges (connections) between nodes.
    min_w : float, optional, default 0
        minimum value for wieights
    max_w : float, optional, default 0
        maximum value for wieights

    Returns
    -------
    graph : dict
        A dictionary representing the graph.
    '''

    nodes      = [f"node_{i}" for i in range(num_nodes)]
    graph      = {node: {} for node in nodes}
    edges      = set()

    while len(edges) < num_edges:
        # Choose randomly two nodes
        a, b = random.sample(nodes, 2)

        if (a, b) not in edges and (b, a) not in edges:
            # If not exist create the link
            weight = random.random()*(max_w - min_w) + min_w
            graph[a][b] = weight
            graph[b][a] = weight
            edges.add((a, b))

    return graph


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
    labels = {k: f"{v:.2f}" for k, v in nx.get_edge_attributes(G, 'weight').items()}

    plt.figure(1)
    nx.draw(G, pos, with_labels=True, node_size=1000,
            node_color="lightblue", edge_color="gray", font_size=10)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

    plt.title("Graph Representation")
    plt.show()


def generate_maze_dfs(rows, cols):
    '''
    Function for generating a maze via Depth-First Search.
    Since the creation algorithm proceeds in steps of 2,
    to avoid a border that is formed by two cells it is
    useful for rows and cols to be odd. 
    At the end of the creation of the maze to create one
    with more solutions we proceed to remove some cells
    between adjacent strees.

    Parameters
    ----------
    rows : int
        number of rows
    cols : int
        number of columns
    
    Returns
    -------
    maze : 2darray
        grid with the maze
    '''
    # Start with all wall, ad set (1, 1) as starting point
    maze = np.ones((rows, cols), dtype=int)
    stack = [(1, 1)]
    maze[1, 1] = 0

    # directions with a size step of 2 to avoid casual wall
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

    while stack:
        r, c = stack[-1]
        random.shuffle(directions)
        found = False

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows-1 and 1 <= nc < cols-1 and maze[nr, nc] == 1:
                maze[nr, nc] = 0  
                maze[r + dr//2, c + dc//2] = 0 
                stack.append((nr, nc))
                found = True
                break
        
        if not found:
            stack.pop()  # Backtracking
    
    add_extra_paths(maze)

    return maze


def add_extra_paths(maze):
    '''
    Funcion for adding new paths

    Parameters
    ----------
    maze : 2darray
        grid with the maze
    '''
    rows, cols = maze.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if maze[r, c] == 1:
                if (maze[r-1, c] == 0 and maze[r+1, c] == 0) or (maze[r, c-1] == 0 and maze[r, c+1] == 0):
                    if random.random() < 0.01:
                        maze[r, c] = 0


def plot_maze(grid, *args, **kwargs):
    '''
    Function to plot the maze ad the found path

    Parameters
    ----------
    grid : 2darray
        maze to plot
    
    args : tuple, optional
        list of all path to plot
    
    Other Parameters
    ----------------
    colors : list
        list of color for different path
    labels : list
        list of labels for different path
    '''

    # Default values
    colors = kwargs.get("colors", plt.cm.jet(np.linspace(0, 1, len(args))))
    labels = kwargs.get("labels", [f'p_{i}' for i in range(len(args))])

    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray_r')
    
    if args:
        for path, c, l in zip(args, colors, labels):

            path_x, path_y = zip(*path)
            ax.plot(path_y, path_x, marker='.', color=c, label=l)
        
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.legend()
    plt.tight_layout()
    plt.show()
