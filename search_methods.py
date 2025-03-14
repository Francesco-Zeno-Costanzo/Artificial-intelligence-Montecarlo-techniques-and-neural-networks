"""
Code that implement several alghoritms for searching
"""
import numpy as np


def build_graph_from_grid(grid):
    '''
    Converts a grid into a graph representation.
    
    Parameters
    ----------
    grid : list
        The grid we want to explore
    
    Returns
    -------
    graph : dict
        grid saved as a graph
    '''
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()

    rows, cols = len(grid), len(grid[0])
    graph      = {}
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:  # Assume 1 is a wall/obstacle
                continue

            neighbors = {}
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    neighbors[(nr, nc)] = 1  # Assume uniform cost
            
            graph[(r, c)] = neighbors
    
    return graph


def Search(struct, src, dst, method='Breadth First', level=None):
    '''
    Function that implements search methods.
    The aviable metods are: 
    1) Breadth First
    2) Depth First
    3) Limited Depth

    Parameters
    ----------
    struct : 2darray or dict
        dictionary created by load_graph function
    src : tuple or str
        name of the source node
    dst : tuple or str
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
    cost : float
        cost of the entire path
    '''
    graph = None
    # If struct is a grid convert it to a graph
    if isinstance(struct, (list, np.ndarray)):
        graph = build_graph_from_grid(struct)
    else :
        # Otherwise struct is already a graph
        graph = struct
        # and src and dst will be string
        # instead of tuple so for case-sensitive:
        src = src.lower()
        dst = dst.lower()

    # Useful variables to switch methods
    idx   = 0 if method == 'Breadth First' else -1
    level = np.inf if method != "Limited Depth" else level
    if method == 'Limited Depth' and level is None:
        raise Exception('For Limited Depth method you must pass a level')
    
    # Variable used in the search
    info    = [(src, [src], 0)]
    visited = {src}

    while info: # Until we can search
        # Unpack the information
        (node, path, cost) = info.pop(idx)

        # For all nearby nodes
        for temp in graph[node].keys():

            # If I find the destination I've finished
            if temp == dst:
                return path + [temp], cost + graph[node][temp]
            
            else:
                # Otherwise, if temp is a new node
                if temp not in visited:
                    # Add temp to visited nodes, to avoid waste of time
                    visited.add(temp)
                    # If I don't exceed the limit I'll continue the search
                    if len(path) < level:
                        info.append((temp, path + [temp], cost + graph[node][temp]))


def iterative_deepening_search(struct, src, dst):
    '''
    Function that implements iterative deepening search method.

    Parameters
    ----------
    struct : 2darray or dict
        dictionary created by load_graph function
    src : tuple or str
        name of the source node
    dst : tuple or str
        name of the destination node
    
    Returns
    -------
    path : list
        list of al node form src to dst
    cost : float
        cost of the entire path
    '''
    
    level = 0  # Depth level
    check = 1  # Flag to control code stopping
    while check == 1:

        level += 1  # Increase the depth
        try : 
            path, cost = Search(struct, src, dst, method='Limited Depth', level=level)
        except TypeError:
            # If path is None level is to small we need to increase it
            continue # Restart the loop
        else :
            # Otherwise we have our result and we can stop
            check = 0
    return path, cost


if __name__ == "__main__":

    from utils import *

    graph = load_graph('data.txt')    
    
    print(Search(graph, "Oradea", "Bucharest"))
    print(iterative_deepening_search(graph, "Oradea", "Bucharest"))
    plot_graph(graph)

    graph = generate_random_graph(16, 22)
    
    print(Search(graph, "Node_2", "Node_7"))
    print(iterative_deepening_search(graph, "Node_2", "Node_7"))
    plot_graph(graph)

