from PIL import Image
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

def euclidean_dist(v1, v2):
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def set_valued_fun_N(M, v):
    row, col = v
    neighbors = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    
    for dr, dc in directions:
        if 0 <= row + dr < M.shape[0] and 0 <= col + dc < M.shape[1] and M[row + dr, col + dc]:
            neighbors.append((row + dr, col + dc))
    return neighbors

def recover_path(s, g, pred):
    path = []
    v = g
    while v != s:
        path.append(v)
        v = pred[v]
    path.append(s)
    return path[::-1]

def a_star_search(M, s, g):
    cost_to = {}
    pred = {}
    est_total_cost = {}

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            cost_to[(r, c)] = float('inf')
            est_total_cost[(r, c)] = float('inf')
    
    cost_to[s] = 0
    est_total_cost[s] = euclidean_dist(s, g)

    Q = []
    heapq.heappush(Q, (euclidean_dist(s, g), s))

    while Q:
        _, v = heapq.heappop(Q)
        if v == g:
            return recover_path(s, g, pred)

        for neighbor in set_valued_fun_N(M, v):
            pvi = cost_to[v] + euclidean_dist(v, neighbor)

            if pvi < cost_to[neighbor]:
                pred[neighbor] = v
                cost_to[neighbor] = pvi
                est_total_cost[neighbor] = pvi + euclidean_dist(neighbor, g)

                if neighbor in [x[1] for x in Q]:
                    for j, (_, item) in enumerate(Q):
                        if item == neighbor:
                            Q[j] = (est_total_cost[neighbor], neighbor)
                            heapq.heapify(Q)
                else:
                    heapq.heappush(Q, (est_total_cost[neighbor], neighbor))
    return None

def sample_vertex(M):
    while True:
        row = np.random.randint(0, M.shape[0])
        col = np.random.randint(0, M.shape[1])
        if M[row, col] == 1:
            return (row, col)

#bresenham algorithm for diagonal lines
def reachable(M, v1, v2):
    r1, c1 = v1
    r2, c2 = v2
    delta_r = r2 - r1
    delta_c = c2 - c1

    dx = abs(delta_r)
    dy = abs(delta_c)
    sx = 1 if r1 < r2 else -1
    sy = 1 if c1 < c2 else -1
    err = dx - dy
    r, c = r1, c1
    while True:
        if M[r, c] == 0:
            return False
        if (r, c) == (r2, c2):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            r += sx
        if e2 < dx:
            err += dx
            c += sy
    return True

def add_vertex_to_prm(M, G, v_new, d_max):
    G.add_node(v_new)
    for v in G.nodes:
        if v != v_new and euclidean_dist(v, v_new) < d_max:
            if reachable(M, v, v_new):
                G.add_edge(v, v_new, weight=euclidean_dist(v, v_new))

def construct_prm(M, N, d_max, s, g):
    G = nx.Graph()
    add_vertex_to_prm(M, G, s, d_max)
    add_vertex_to_prm(M, G, g, d_max)
    for _ in range(N):
        v_new = sample_vertex(M)
        add_vertex_to_prm(M, G, v_new, d_max)
    return G

if __name__ == "__main__":
    occupancy_map_img = Image.open('occupancy_map.png')
    M = (np.asarray(occupancy_map_img) > 0).astype(int)

    # Route Planning with A* Search
    s = (635, 140)
    g = (350, 400)

    path = a_star_search(M, s, g)
    print(f"Total A* path length: {len(path)}")

    plt.imshow(M, cmap=ListedColormap(['white', 'black']))
    plt.scatter(s[1], s[0], color='red')
    plt.scatter(g[1], g[0], color='blue')
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], color='green')
    plt.show()

    # Route Planning with Probabilistic Roadmaps (PRM)
    N = 2500
    d_max = 75

    G = construct_prm(M, N, d_max, s, g)
    print(f"Number of nodes in PRM: {G.number_of_nodes()}")

    path = nx.astar_path(G, s, g, heuristic=euclidean_dist, weight='weight')
    print(f"Total PRM-based path length: {len(path)}")

    plt.imshow(M, cmap=ListedColormap(['white', 'black']))
    pos = {v: (v[1], v[0]) for v in G.nodes}
    nx.draw(G, pos, node_size=1, edge_color='red', node_color='blue')
    plt.scatter(s[1], s[0], color='red')
    plt.scatter(g[1], g[0], color='blue')
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], color='green')
    plt.show()
