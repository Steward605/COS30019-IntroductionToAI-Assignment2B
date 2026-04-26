from collections import deque

def breadth_first_search(start_node, goal_nodes, graph, debug=False):
    goal_nodes = set(goal_nodes)
    
    # Acts as a FIFO queue, so BFS expands the shallowest node first
    frontier = deque([(start_node, [start_node])])
    
    # EXPLORED set stores nodes that have already been expanded
    explored = set()
    number_of_nodes_created = 1

    while frontier:
        # pop shallowest node in the queue
        current_node, current_path = frontier.popleft()

        # Stop as soon as a goal node is reached
        if current_node in goal_nodes:
            return current_node, number_of_nodes_created, current_path

        explored.add(current_node)
        outgoing_edges = graph.get(current_node, [])
        
        # discovered nodes are sorted in ascending node ID before being appended to frontier
        neighbor_nodes = sorted(neighbor for neighbor, cost in outgoing_edges) 

        # Avoid inserting duplicate states that are already waiting in the frontier
        frontier_states = {node for node, _ in frontier}

        for neighbor_node in neighbor_nodes:
            if neighbor_node not in explored and neighbor_node not in frontier_states:
                frontier.append((neighbor_node, current_path + [neighbor_node]))
                number_of_nodes_created += 1

        if debug:
            print(f"Current node: {current_node} | Frontier: {[node for node, _ in frontier]} | Explored: {sorted(explored)}")

    # Failure: no goal is reachable if the frontier becomes empty
    return None, number_of_nodes_created, []