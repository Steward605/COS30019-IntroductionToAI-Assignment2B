from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from travel_time import calculate_travel_time_from_traffic_flow
from algorithms.astar import a_star_search
from algorithms.bfs import breadth_first_search
from algorithms.dfs import depth_first_search
from algorithms.gbfs import greedy_best_first_search
from algorithms.cus1 import bs_search
from algorithms.cus2 import ida_star_search

ROUTE_GRAPH_FOLDER = Path("route_graph_data")
SCATS_NODES_FILE = ROUTE_GRAPH_FOLDER / "scats_nodes.csv"
SCATS_EDGES_FILE = ROUTE_GRAPH_FOLDER / "scats_edges_prepared.csv"
A2A_NODE_POSITIONS_FILE = ROUTE_GRAPH_FOLDER / "a2a_node_positions.json"
PROCESSED_TRAFFIC_FILE = Path("traffic_flow_model_data/processed_hourly_traffic_flow.csv")
SCALER_FILE = Path("traffic_flow_model_data/traffic_flow_scaler.pkl")
MODEL_FILES = {"gru": Path("models/gru_traffic_model.keras"),"lstm": Path("models/lstm_traffic_model.keras")}
SUPPORTED_ALGORITHMS = ["bfs", "dfs", "gbfs", "astar", "cus1", "cus2"]
FLOW_SOURCE = "destination"  # travel time from SCATS site A to B approximated using the hourly volume at SCATS site B, options: "destination", "start"
OUTPUT_ROUTE_RESULTS_FILE = Path("route_results.csv")

# LOAD FILES
# ============================================================
def load_scats_nodes(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run prepare_scats_graph.py first.")
    nodes_df = pd.read_csv(file_path)
    if "scats_site" not in nodes_df.columns:
        raise ValueError("scats_nodes.csv must contain scats_site.")
    nodes_df["scats_site"] = pd.to_numeric(nodes_df["scats_site"], errors="coerce")
    nodes_df = nodes_df.dropna(subset=["scats_site"]).copy()
    nodes_df["scats_site"] = nodes_df["scats_site"].astype(int)
    return nodes_df

def load_scats_edges(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run prepare_scats_graph.py first.")
    edges_df = pd.read_csv(file_path)
    required_columns = {"start_scats", "end_scats", "distance_km"}
    if not required_columns.issubset(edges_df.columns):
        raise ValueError("scats_edges_prepared.csv must contain start_scats, end_scats, and distance_km.")
    edges_df["start_scats"] = pd.to_numeric(edges_df["start_scats"], errors="coerce")
    edges_df["end_scats"] = pd.to_numeric(edges_df["end_scats"], errors="coerce")
    edges_df["distance_km"] = pd.to_numeric(edges_df["distance_km"], errors="coerce")
    edges_df = edges_df.dropna(subset=["start_scats", "end_scats", "distance_km"]).copy()
    edges_df["start_scats"] = edges_df["start_scats"].astype(int)
    edges_df["end_scats"] = edges_df["end_scats"].astype(int)
    edges_df = edges_df[edges_df["distance_km"] > 0].copy()
    if edges_df.empty:
        raise ValueError("No valid SCATS edges found.")
    return edges_df.reset_index(drop=True)

def load_a2a_node_positions(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run prepare_scats_graph.py first.")
    with open(file_path, "r", encoding="utf-8") as file:
        raw_positions = json.load(file)
    node_positions = {}
    for node, position in raw_positions.items():
        node_positions[int(node)] = (
            float(position[0]),  # longitude
            float(position[1])   # latitude
        )
    return node_positions

def load_hourly_traffic_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run data-process.py first.")
    traffic_df = pd.read_csv(file_path)
    required_columns = {"scats_site_number", "timestamp", "traffic_flow"}
    if not required_columns.issubset(traffic_df.columns):
        raise ValueError("processed_hourly_traffic_flow.csv must contain scats_site_number, timestamp, and traffic_flow.")
    traffic_df["scats_site_number"] = pd.to_numeric(traffic_df["scats_site_number"], errors="coerce")
    traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"], errors="coerce")
    traffic_df["traffic_flow"] = pd.to_numeric(traffic_df["traffic_flow"], errors="coerce")
    traffic_df = traffic_df.dropna(subset=["scats_site_number", "timestamp", "traffic_flow"]).copy()
    traffic_df["scats_site_number"] = traffic_df["scats_site_number"].astype(int)
    return traffic_df.sort_values(["scats_site_number", "timestamp"]).reset_index(drop=True)

def load_model_and_scaler(model_type):
    if model_type not in MODEL_FILES:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}. Use one of: {list(MODEL_FILES.keys())}")
    model_file = MODEL_FILES[model_type]
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}. Run the training script first.")
    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}. Run data-process.py first.")
    model = tf.keras.models.load_model(model_file, compile=False)
    with open(SCALER_FILE, "rb") as file:
        scaler = pickle.load(file)
    return model, scaler

# TRAFFIC FLOW PREDICTION
# ============================================================
def get_fallback_flow_for_site(traffic_df, scats_site, departure_hour):
    site_data = traffic_df[traffic_df["scats_site_number"] == scats_site].copy()
    if site_data.empty:
        return float(traffic_df["traffic_flow"].median())
    same_hour_data = site_data[site_data["timestamp"].dt.hour == departure_hour.hour]
    if not same_hour_data.empty:
        return float(same_hour_data["traffic_flow"].median())
    return float(site_data["traffic_flow"].median())

def predict_flow_for_site(model, scaler, traffic_df, scats_site, departure_datetime):
    departure_hour = pd.Timestamp(departure_datetime).floor("h")
    site_data = traffic_df[traffic_df["scats_site_number"] == scats_site].sort_values("timestamp")
    previous_12_hours = site_data[(site_data["timestamp"] >= departure_hour - pd.Timedelta(hours=12)) &(site_data["timestamp"] < departure_hour)]
    if len(previous_12_hours) < 12:
        return get_fallback_flow_for_site(traffic_df, scats_site, departure_hour)
    previous_values = previous_12_hours.tail(12)[["traffic_flow"]].values
    scaled_values = scaler.transform(previous_values)
    model_input = scaled_values.reshape(1, 12, 1)
    scaled_prediction = model.predict(model_input, verbose=0)
    predicted_flow = scaler.inverse_transform(scaled_prediction)[0][0]
    return max(0.0, float(predicted_flow))

def predict_flows_for_graph_sites(model, scaler, traffic_df, graph_sites, departure_datetime):
    predicted_flows = {}
    for scats_site in sorted(graph_sites):
        predicted_flows[scats_site] = predict_flow_for_site(model=model,scaler=scaler,traffic_df=traffic_df,scats_site=scats_site,departure_datetime=departure_datetime)
    return predicted_flows

# BUILD TRAVEL-TIME GRAPH
# ============================================================
def choose_flow_site(start_scats, end_scats):
    if FLOW_SOURCE == "destination":
        return end_scats
    if FLOW_SOURCE == "start":
        return start_scats
    raise ValueError("FLOW_SOURCE must be either 'destination' or 'start'.")

def build_edges_with_travel_time(edges_df, predicted_flows):
    rows = []
    for _, row in edges_df.iterrows():
        start_scats = int(row["start_scats"])
        end_scats = int(row["end_scats"])
        distance_km = float(row["distance_km"])
        flow_site = choose_flow_site(start_scats, end_scats)
        if flow_site not in predicted_flows:
            raise KeyError(f"No predicted traffic flow found for SCATS site {flow_site}.")
        predicted_flow = predicted_flows[flow_site]
        travel_time_minutes = calculate_travel_time_from_traffic_flow(predicted_flow=predicted_flow,distance_km=distance_km,is_congested=False)
        rows.append({
            "start_scats": start_scats,
            "end_scats": end_scats,
            "distance_km": distance_km,
            "flow_site": flow_site,
            "predicted_flow": round(predicted_flow, 2),
            "travel_time_minutes": travel_time_minutes
        })
    return pd.DataFrame(rows)

def build_a2a_travel_time_graph(edges_with_time_df):
    graph = {}
    for _, row in edges_with_time_df.iterrows():
        start_scats = int(row["start_scats"])
        end_scats = int(row["end_scats"])
        travel_time_minutes = float(row["travel_time_minutes"])
        graph.setdefault(start_scats, [])
        graph.setdefault(end_scats, [])
        graph[start_scats].append((end_scats, travel_time_minutes))
    for node in graph:
        graph[node] = sorted(graph[node], key=lambda item: item[0])
    return graph

def calculate_path_cost(graph, path):
    total_cost = 0.0
    for start_node, end_node in zip(path[:-1], path[1:]):
        edge_found = False
        for neighbour_node, edge_cost in graph.get(start_node, []):
            if neighbour_node == end_node:
                total_cost += float(edge_cost)
                edge_found = True
                break
        if not edge_found:
            raise ValueError(f"No edge found from {start_node} to {end_node}.")
    return round(total_cost, 2)

def run_assignment_2a_algorithm(algorithm_name, graph, node_positions, origin_scats, destination_scats):
    goal_nodes = [destination_scats]
    if algorithm_name == "bfs":
        goal_node, nodes_created, path = breadth_first_search(
            start_node=origin_scats,
            goal_nodes=goal_nodes,
            graph=graph,
            debug=False
        )
    elif algorithm_name == "dfs":
        goal_node, nodes_created, path = depth_first_search(
            start_node=origin_scats,
            goal_nodes=goal_nodes,
            graph=graph,
            debug=False
        )
    elif algorithm_name == "gbfs":
        goal_node, nodes_created, path = greedy_best_first_search(
            start_node=origin_scats,
            goal_nodes=goal_nodes,
            graph=graph,
            node_positions=node_positions,
            debug=False
        )
    elif algorithm_name == "astar":
        goal_node, nodes_created, path = a_star_search(
            start_node=origin_scats,
            goal_nodes=goal_nodes,
            graph=graph,
            node_positions=node_positions,
            debug=False
        )
    elif algorithm_name == "cus1":
        goal_node, nodes_created, path = bs_search(
            node_positions=node_positions,
            edges=graph,
            origin=origin_scats,
            destinations=goal_nodes
        )
    elif algorithm_name == "cus2":
        goal_node, nodes_created, path = ida_star_search(
            start_node=origin_scats,
            goal_nodes=goal_nodes,
            graph=graph,
            node_positions=node_positions,
            debug=False
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    if goal_node is None or not path:
        return {
            "algorithm": algorithm_name,
            "found": False,
            "path": [],
            "travel_time_minutes": None,
            "nodes_created": nodes_created,
            "message": "No route found."
        }
    travel_time_minutes = calculate_path_cost(graph, path)
    return {
        "algorithm": algorithm_name,
        "found": True,
        "path": path,
        "travel_time_minutes": travel_time_minutes,
        "nodes_created": nodes_created,
        "message": "Route found."
    }
    
def run_all_assignment_2a_algorithms(graph, node_positions, origin_scats, destination_scats, algorithm_names=None):
    if algorithm_names is None:
        algorithm_names = SUPPORTED_ALGORITHMS
    results = {}
    for algorithm_name in algorithm_names:
        results[algorithm_name] = run_assignment_2a_algorithm(algorithm_name=algorithm_name,graph=graph,node_positions=node_positions,origin_scats=origin_scats,destination_scats=destination_scats)
    return results

def build_networkx_graph(edges_with_time_df):
    graph = nx.DiGraph()
    for _, row in edges_with_time_df.iterrows():
        start_scats = int(row["start_scats"])
        end_scats = int(row["end_scats"])
        graph.add_edge(start_scats,end_scats,distance_km=float(row["distance_km"]),travel_time_minutes=float(row["travel_time_minutes"]),predicted_flow=float(row["predicted_flow"]),flow_site=int(row["flow_site"]))
    return graph

# TOP-K ROUTES USING NETWORKX
# ===========================
def calculate_networkx_route_values(graph, route):
    total_distance = 0.0
    total_travel_time = 0.0
    for start_node, end_node in zip(route[:-1], route[1:]):
        edge_data = graph[start_node][end_node]
        total_distance += float(edge_data["distance_km"])
        total_travel_time += float(edge_data["travel_time_minutes"])
    return total_distance, total_travel_time

def find_top_k_routes(graph, origin_scats, destination_scats, top_k):
    routes = []
    try:
        route_generator = nx.shortest_simple_paths(graph,source=origin_scats,target=destination_scats,weight="travel_time_minutes")
        for route in route_generator:
            total_distance, total_travel_time = calculate_networkx_route_values(graph, route)
            routes.append({
                "route_number": len(routes) + 1,
                "path": route,
                "distance_km": round(total_distance, 4),
                "travel_time_minutes": round(total_travel_time, 2)
            })
            if len(routes) >= top_k:
                break
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []
    return routes

# OUTPUT HELPERS
# ==============
def format_path(path):
    return " -> ".join(str(node) for node in path)

def get_location_lookup(nodes_df):
    if "location_description" not in nodes_df.columns:
        return {}
    return {
        int(row["scats_site"]): str(row["location_description"])
        for _, row in nodes_df.iterrows()
    }

def print_route_details(route, location_lookup):
    print(f"Route {route['route_number']}")
    print(f"Path: {format_path(route['path'])}")
    print(f"Estimated travel time: {route['travel_time_minutes']:.2f} minutes")
    print(f"Estimated distance: {route['distance_km']:.2f} km")
    print("Route intersections:")
    for scats_site in route["path"]:
        location = location_lookup.get(scats_site, "Unknown location")
        print(f"  {scats_site}: {location}")
    print()

def save_route_results(algorithm_results, routes):
    rows = []
    for algorithm_name, algorithm_result in algorithm_results.items():
        rows.append({
            "source": "ASSIGNMENT_2A_ALGORITHM",
            "algorithm": algorithm_name.upper(),
            "route_number": 1,
            "path": format_path(algorithm_result["path"]) if algorithm_result["found"] else "",
            "travel_time_minutes": algorithm_result["travel_time_minutes"],
            "distance_km": "",
            "nodes_created": algorithm_result["nodes_created"],
            "message": algorithm_result["message"]
        })
    for route in routes:
        rows.append({
            "source": "NETWORKX_TOP_K",
            "algorithm": "NETWORKX",
            "route_number": route["route_number"],
            "path": format_path(route["path"]),
            "travel_time_minutes": route["travel_time_minutes"],
            "distance_km": route["distance_km"],
            "nodes_created": "",
            "message": "Top-k route"
        })
    pd.DataFrame(rows).to_csv(OUTPUT_ROUTE_RESULTS_FILE, index=False)

# This function is used by the GUI later.
def find_routes(origin_scats, destination_scats, departure_datetime, model_type="gru", top_k_routes=5, algorithm_names=None):
    nodes_df = load_scats_nodes(SCATS_NODES_FILE)
    edges_df = load_scats_edges(SCATS_EDGES_FILE)
    node_positions = load_a2a_node_positions(A2A_NODE_POSITIONS_FILE)
    origin_scats = int(origin_scats)
    destination_scats = int(destination_scats)
    available_scats_sites = set(nodes_df["scats_site"].tolist())
    if origin_scats not in available_scats_sites:
        raise ValueError(f"Origin SCATS {origin_scats} is not available in scats_nodes.csv.")
    if destination_scats not in available_scats_sites:
        raise ValueError(f"Destination SCATS {destination_scats} is not available in scats_nodes.csv.")
    traffic_df = load_hourly_traffic_data(PROCESSED_TRAFFIC_FILE)
    model, scaler = load_model_and_scaler(model_type)
    graph_sites = set(edges_df["start_scats"].tolist()) | set(edges_df["end_scats"].tolist())
    predicted_flows = predict_flows_for_graph_sites(model=model,scaler=scaler,traffic_df=traffic_df,graph_sites=graph_sites,departure_datetime=departure_datetime)
    edges_with_time_df = build_edges_with_travel_time(edges_df, predicted_flows)
    a2a_graph = build_a2a_travel_time_graph(edges_with_time_df)
    algorithm_results = run_all_assignment_2a_algorithms(graph=a2a_graph,node_positions=node_positions,origin_scats=origin_scats,destination_scats=destination_scats,algorithm_names=algorithm_names)
    networkx_graph = build_networkx_graph(edges_with_time_df)
    top_k_routes = find_top_k_routes(graph=networkx_graph,origin_scats=origin_scats,destination_scats=destination_scats,top_k=top_k_routes)
    location_lookup = get_location_lookup(nodes_df)

    return {
        "origin_scats": origin_scats,
        "destination_scats": destination_scats,
        "departure_datetime": departure_datetime,
        "model_type": model_type,
        "algorithm_results": algorithm_results,
        "top_k_routes": top_k_routes,
        "location_lookup": location_lookup,
        "edges_with_time": edges_with_time_df
    }

# This is only for testing route_engine.py directly in the terminal.
def main():
    result = find_routes(origin_scats=2000,destination_scats=3002,departure_datetime="2006-10-15 15:00:00",model_type="gru",top_k_routes=5)
    print()
    print("========== ROUTE ENGINE RESULT ==========")
    print(f"Model used: {result['model_type'].upper()}")
    print(f"Origin SCATS: {result['origin_scats']}")
    print(f"Destination SCATS: {result['destination_scats']}")
    print(f"Departure datetime: {result['departure_datetime']}")
    print()
    print("Assignment 2A algorithm results:")
    algorithm_results = result["algorithm_results"]
    for algorithm_name, algorithm_result in algorithm_results.items():
        print()
        print(f"{algorithm_name.upper()} result:")
        if algorithm_result["found"]:
            print(f"Path: {format_path(algorithm_result['path'])}")
            print(f"Estimated travel time: {algorithm_result['travel_time_minutes']:.2f} minutes")
            print(f"Nodes created: {algorithm_result['nodes_created']}")
        else:
            print(algorithm_result["message"])
    print()
    print("Top-k routes:")
    top_k_routes = result["top_k_routes"]
    location_lookup = result["location_lookup"]
    if not top_k_routes:
        print("No top-k routes found.")
    else:
        for route in top_k_routes:
            print_route_details(route, location_lookup)
    save_route_results(algorithm_results, top_k_routes)
    print(f"Route results saved to: {OUTPUT_ROUTE_RESULTS_FILE}")

if __name__ == "__main__":
    main()