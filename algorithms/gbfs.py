from heapq import heappush, heappop
from .utils import heuristic

def greedy_best_first_search(start_node, goal_nodes, graph, node_positions, debug=False):
    goal_nodes = set(goal_nodes)
    
    # frontier is a priority queue,  node with smallest heuristic value is expanded first
    frontier = []
    
    # Stores nodes that have already been expanded
    explored = set()
    number_of_nodes_created = 1
    insertion_order = 0

    start_heuristic = heuristic(start_node, goal_nodes, node_positions)
    heappush(frontier, (start_heuristic, start_node, insertion_order, [start_node]))

    while frontier:
        # pop node with smallest heuristic value,
        # if tied, smaller node ID is preferred
        _, current_node, _, current_path = heappop(frontier)
        
        if current_node in goal_nodes:
            return current_node, number_of_nodes_created, current_path
        
        # Skip this entry if the node has already been expanded earlier
        if current_node in explored:
            continue
        explored.add(current_node)
        outgoing_edges = graph.get(current_node, [])
        
        # Neighbours are generated in ascending node ID order
        neighbor_nodes = sorted(neighbor for neighbor, cost in outgoing_edges)
        
        # Avoid inserting duplicate states that are already waiting in the frontier
        frontier_states = {node for _, node, _, _ in frontier}
        for neighbor_node in neighbor_nodes:
            if neighbor_node not in explored and neighbor_node not in frontier_states:
                insertion_order += 1
                neighbor_heuristic = heuristic(neighbor_node, goal_nodes, node_positions)
                heappush(frontier, (neighbor_heuristic, neighbor_node, insertion_order, current_path + [neighbor_node]))
                number_of_nodes_created += 1
        if debug:
            print(f"Current: {current_node}, " f"Frontier: {[node for _, node, _, _ in frontier]}, " f"Explored: {sorted(explored)}")
            
    # Failure: no goal is reachable if the frontier becomes empty
    return None, number_of_nodes_created, []