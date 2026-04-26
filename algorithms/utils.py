import math

def read_route_problem(file_path):
    """
    Reads a route-finding problem from a text file.

    Returns:
        node_positions: Maps each node ID to its (x, y) coordinates.
        graph: Adjacency list mapping each node to its outgoing edges.
        origin_node: The start node.
        destination_nodes: The goal nodes.
    """
    node_positions = {}
    graph = {}
    origin_node = None
    destination_nodes = []
    current_section = None

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Identify which part of the problem specification is being read
            if line in {"Nodes:", "Edges:", "Origin:", "Destinations:"}:
                current_section = line[:-1].lower()
                continue

            if current_section == "nodes":
                node_id_text, coordinates_text = line.split(":")
                node_id = int(node_id_text)
                x_coordinate, y_coordinate = map(
                    int, coordinates_text.strip()[1:-1].split(",")
                )

                # Store each node's coordinates and ensure it exists in the graph
                node_positions[node_id] = (x_coordinate, y_coordinate)
                graph.setdefault(node_id, [])

            elif current_section == "edges":
                nodes_text, cost_text = line.split(":")
                start_node, end_node = map(int, nodes_text.strip()[1:-1].split(","))
                edge_cost = int(cost_text)

                graph.setdefault(start_node, [])
                graph.setdefault(end_node, [])
                graph[start_node].append((end_node, edge_cost))

            elif current_section == "origin":
                origin_node = int(line)

            elif current_section == "destinations":
                destination_nodes = [int(node) for node in line.split(";")]

    # The problem file must define both an origin and at least one destination
    if origin_node is None:
        raise ValueError("Origin node is missing.")

    if not destination_nodes:
        raise ValueError("Destination nodes are missing.")

    return node_positions, graph, origin_node, destination_nodes

# Returns the straight-line estimate from a node to the nearest goal node
def heuristic(node, goal_nodes, node_positions):
    if node not in node_positions:
        return 0
    node_x, node_y = node_positions[node]
    return min(
        (math.sqrt((node_x - node_positions[goal][0]) ** 2 + (node_y - node_positions[goal][1]) ** 2) for goal in goal_nodes if goal in node_positions),
        default=0
    )