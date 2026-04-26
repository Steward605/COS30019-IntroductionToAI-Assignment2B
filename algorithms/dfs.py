def depth_first_search(start_node, goal_nodes, graph, debug=False):
    goal_nodes = set(goal_nodes)
    
    # frontier is a LIFO stack, so most recently generated node is expanded first
    frontier = [(start_node, [start_node])]
    
    # Stores nodes that have already been expanded
    explored = set()
    number_of_nodes_created = 1

    while frontier:
        current_node, current_path = frontier.pop()
    
        # Stop as soon as a goal node is reached
        if current_node in goal_nodes:
            return current_node, number_of_nodes_created, current_path

        explored.add(current_node)
        outgoing_edges = graph.get(current_node, [])
        
        # Neighbours are processed in reverse node ID order before being pushed,
        # so that the smaller node ID is expanded first when popped from the stack
        neighbor_nodes = sorted(
            (neighbor for neighbor, cost in outgoing_edges),
            reverse=True
        )

        # Avoid inserting duplicate states that are already waiting in the frontier
        frontier_states = {node for node, _ in frontier}

        for neighbor_node in neighbor_nodes:
            if neighbor_node not in explored and neighbor_node not in frontier_states:
                frontier.append((neighbor_node, current_path + [neighbor_node]))
                number_of_nodes_created += 1

        if debug:
            print(f"Current: {current_node}, Frontier: {[n for n,_ in frontier]}")

    # Failure: no goal is reachable if the frontier becomes empty
    return None, number_of_nodes_created, []