import heapq
import numpy as np

class AStar:

    def __init__(self, struct, h_func=None):
        '''
        Initialize A* with a graph or grid.

        Parameters
        ----------
        struct : dict or list
            If it is a dictionary -> it is treated as a graph.
            If it is a list of lists -> it is treated as a grid.
        h_func : callable, optional
            Heuristic function h(n), default Manhattan or Dijkstra distance is used.
        '''
        self.is_grid = isinstance(struct, (list, np.ndarray))
        
        if self.is_grid:
            self.struct = self.build_graph_from_grid(struct)
        else:
            self.struct = struct
        
        self.h_func = h_func if h_func is not None else self.default_heuristic


    def build_graph_from_grid(self, grid):
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


    def default_heuristic(self, n1, n2):
        '''
        Default heuristic: Manhattan distance for grids, 0 for graphs.

        Parameters
        ----------
        n1 : tuple
            Actual node
        n2 : tuple
            Goal node
        '''
        if self.is_grid:
            x1, y1 = n1
            x2, y2 = n2
            return  abs(x1 - x2) + abs(y1 - y2) 
        return 0 


    def search(self, start, goal):
        '''
        Executes A* search and returns the found path.

        Parameters
        ----------
        start : tuple or str
            Intial node
        goal : tuple or str
            Final node
        
        Return
        ------
        path : list
            list of al node form start to goal
        current_f : float
            cost of the entire path
        len(visited) : int
            Number of visited Node
        '''
        
        if start not in self.struct or goal not in self.struct:
            return None, float('inf')
        
        visited = {}                                            # All visited nodes for reconstruct the path
        g_cost  = {node:float('inf') for node in self.struct}   # Store cost from start to the actual node
        f_cost  = {node:float('inf') for node in self.struct}   # Store cost from start to goal passing from goal

        g_cost[start] = 0
        f_cost[start] = self.h_func(start, goal)

        # Priority queue created with heap
        prio_queue = []
        heapq.heappush(prio_queue, (f_cost[start], start))

        while prio_queue:

            current_f, current = heapq.heappop(prio_queue)    # Node with the smaller value of f(n)

            if current == goal:
                # If I find the destination I've finished.
                # So from all visited node we reconstruct the path
                path = []
                while current in visited:
                    path.append(current)
                    current = visited[current]
                path.append(start)

                return path[::-1], current_f, len(visited)
            
            # Otherwise search for the neighbors
            for neighbor, cost in self.struct[current].items():
                # Compute cost to arrive here
                tmp_g_cost = g_cost[current] + cost

                # If it is less than the previous value
                if tmp_g_cost < g_cost[neighbor]:
                    # Update
                    visited[neighbor] = current
                    g_cost[neighbor]  = tmp_g_cost
                    f_cost[neighbor]  = tmp_g_cost + self.h_func(neighbor, goal)
                    heapq.heappush(prio_queue, (f_cost[neighbor], neighbor))


if __name__ == "__main__":
    
    from collections import deque
    from search_methods import Search
    from utils import generate_random_graph, generate_maze_dfs, plot_maze
    

    def bfs_distances(graph, goal):
        '''Precompute BFS distances from the goal to all nodes.'''
        distances = {node: float('inf') for node in graph}
        queue = deque([(goal, 0)])
        distances[goal] = 0

        while queue:
            node, dist = queue.popleft()
            for neighbor in graph[node]:
                if distances[neighbor] == float('inf'):  # Not visited
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return distances

    def heuristic(n1, n2):
        return heuristic_values.get(n1, float('inf'))
    
    #==============================================================#
    # First test with random graph                                 #
    #==============================================================#

    start = "node_2"
    goal  = "node_7"

    graph = generate_random_graph(10000, 22000)
    path1, cost1 = Search(graph, start, goal)

    heuristic_values = bfs_distances(graph, 'node_7')
    astar_graph_0 = AStar(graph)
    astar_graph_1 = AStar(graph, h_func=heuristic)


    path2, cost2, nv2 = astar_graph_0.search(start, goal)
    path3, cost3, nv3 = astar_graph_1.search(start, goal)
    print('=====================================================================================')
    print(path1)
    print(path2)
    print(path3)
    print(f"path length of {len(path1)}, with total cost of {cost1:.2f}")
    print(f"path length of {len(path2)}, with total cost of {cost2:.2f}, and {nv2} explored node")
    print(f"path length of {len(path3)}, with total cost of {cost3:.2f}, and {nv3} explored node")
   
    #==============================================================#
    # Second test with a grid                                      #
    #==============================================================#

    grid  = generate_maze_dfs(101, 101)
    start = (1, 1)
    goal  = (99, 99)
    astar = AStar(grid)

    path1, cost1, nv = astar.search(start, goal)
    path2, cost2     = Search(grid, start, goal, method='Depth First')
    print('=====================================================================================')
    print(f"path length {len(path1)}, with total cost of {cost1}")
    print(f"path length {len(path2)}, with total cost of {cost2}")

    plot_maze(grid, path1, path2, labels=[f"A*, cost={cost1}", f'Depth first, cost={cost2}'])

    