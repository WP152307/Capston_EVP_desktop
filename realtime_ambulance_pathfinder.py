import networkx as nx
import random

def create_graph():
    # 주어진 노드 리스트(nx)
    nodes_list = [
        (440, 317), #1
        (822, 311), #2
        (1360, 300), #3
        (425, 641), #4
        (822, 637), #5
        (1090, 638), #6
        (1358, 638), #7
        (422, 974), #8
        (817, 983), #9
        (1090, 983), #10
        (1371, 993), #11
    ]

    # 그래프 생성
    G = nx.Graph()

    # 노드 추가
    for idx, node in enumerate(nodes_list):
        G.add_node(idx, pos=node)

    # 엣지 추가
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 6), (3, 4), (4, 5), (5, 6),
            (3, 7), (4, 8), (5, 9), (6, 10), (7, 8), (8, 9), (9, 10)]
    G.add_edges_from(edges)

    return G

G = create_graph()

def return_shortest_path(ambulance_node_number):
    start_node = ambulance_node_number
    end_node = 2  # 도착 노드를 2으로 고정(병원)
    while start_node == end_node:
        end_node = random.choice(list(G.nodes()))

    shortest_path = nx.shortest_path(G, source=start_node, target=end_node)
    total_distance = sum([G[shortest_path[i]][shortest_path[i+1]].get('weight', 1.0) for i in range(len(shortest_path)-1)])

    shortest_path_str = [str(node) for node in shortest_path]

    '''
    print(f"최단경로: {' -> '.join(shortest_path_str)}")
    print(f"총 이동거리: {int(total_distance)}")
    '''

    return shortest_path