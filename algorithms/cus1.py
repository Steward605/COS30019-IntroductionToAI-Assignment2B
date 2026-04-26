from collections import deque, defaultdict

def bs_search(node_positions, edges, origin, destinations):
    # Goal test: return immediately if the origin is already a destination
    if origin in destinations:
        return origin, 1, [origin]

    # Build reverse edges so the backward search can expand predecessor nodes.
    # This is necessary for graphs that contain directed edges
    reverse_edges = defaultdict(list)
    for src, neighbors in edges.items():
        for dest, cost in neighbors:
            reverse_edges[dest].append((src, cost))

    # Reverse neighbours are sorted once so expansion follows ascending node ID order
    for node in reverse_edges:
        reverse_edges[node].sort(key=lambda x: x[0])

    # Forward frontier, visited set, and paths from the origin
    f_queue = deque([origin])
    f_visited = {origin}
    f_paths = {origin: [origin]}

    # Backward frontier, visited set, and paths to a destination
    sorted_destinations = sorted(destinations)
    b_queue = deque(sorted_destinations)
    b_visited = set(sorted_destinations)
    b_paths = {dest: [dest] for dest in sorted_destinations}

    # Count the number of nodes created
    nodes_created = 1 + len(sorted_destinations)
    
    def expand_forward_layer():
        nonlocal nodes_created
        layer_size = len(f_queue)

        # Expand one forward search layer
        for _ in range(layer_size):
            current = f_queue.popleft()
            current_path = f_paths[current]

            # Expand successors in ascending node ID order
            for neighbor, cost in sorted(edges.get(current, []), key=lambda x: x[0]):
                # Skip repeated states already reached by the forward search
                if neighbor in f_visited:
                    continue
                f_visited.add(neighbor)
                new_path = current_path + [neighbor]
                f_paths[neighbor] = new_path
                f_queue.append(neighbor)
                nodes_created += 1

                # Solution is found when forward search reaches a node that is already reached by backward search
                if neighbor in b_paths:
                    full_path = new_path + b_paths[neighbor][1:]
                    
                    # Return the destination node, not just the meeting node
                    goal_reached = full_path[-1]
                    return goal_reached, full_path
        return None, None


    def expand_backward_layer():
        nonlocal nodes_created
        layer_size = len(b_queue)

        # Expand one backward search layer
        for _ in range(layer_size):
            current = b_queue.popleft()
            current_path = b_paths[current]

            # Expand predecessors in ascending node ID order
            for predecessor, cost in reverse_edges.get(current, []):
                # Skip repeated states already reached by the backward search
                if predecessor in b_visited:
                    continue
                b_visited.add(predecessor)
                new_path = [predecessor] + current_path
                b_paths[predecessor] = new_path
                b_queue.append(predecessor)
                nodes_created += 1

                # A solution is found when the two search frontiers meet
                if predecessor in f_paths:
                    full_path = f_paths[predecessor] + new_path[1:]
                    goal_reached = full_path[-1]
                    return goal_reached, full_path
        return None, None

    # Alternate between forward and backward expansion, one layer at a time
    while f_queue and b_queue:
        meet_node, path = expand_forward_layer()
        if meet_node is not None:
            return meet_node, nodes_created, path
        meet_node, path = expand_backward_layer()
        if meet_node is not None:
            return meet_node, nodes_created, path

    # Failure: no destination node is reachable if the two searches never meet
    return None, nodes_created, []