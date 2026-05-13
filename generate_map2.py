from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import folium
import random
from config_loader import load_config, get_config_value

CONFIG = load_config()
SCATS_NODES_FILE = Path(get_config_value(CONFIG, ["paths", "scats_nodes_file"]))
SCATS_EDGES_FILE = Path(get_config_value(CONFIG, ["paths", "scats_edges_file"]))
ORIGIN_COLOR = get_config_value(CONFIG, ["visual", "origin_color"], "#4da3ff")
DESTINATION_COLOR = get_config_value(CONFIG, ["visual", "destination_color"], "#4cd964")
ROUTE_COLOR = get_config_value(CONFIG, ["visual", "route_color"], "#ffb000")

nodes_df = pd.read_csv(SCATS_NODES_FILE)
edges_df = pd.read_csv(SCATS_EDGES_FILE)

node_coord_lookup = {}
for _, row in nodes_df.iterrows():
    site = int(row["scats_site"])
    node_coord_lookup[site] = {
        "latitude":  float(row["latitude"])  + random.uniform(-0.0004, 0.0004),
        "longitude": float(row["longitude"]) + random.uniform(-0.0004, 0.0004),
    }

# -----------------------------------------------
# LOAD SCALER AND HISTORICAL TRAFFIC DATA
# -----------------------------------------------
with open("traffic_flow_model_data/traffic_flow_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

hourly_df = pd.read_csv("traffic_flow_model_data/processed_hourly_traffic_flow.csv")
hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
hourly_df["hour"] = hourly_df["timestamp"].dt.hour

# -----------------------------------------------
# LOAD THE ROUTE FROM route_results.csv
# -----------------------------------------------
ROUTE_RESULTS_FILE = Path("route_results.csv")

route_sites = []  # This will hold the list of SCATS sites in the route

if ROUTE_RESULTS_FILE.exists():
    results_df = pd.read_csv(ROUTE_RESULTS_FILE)
    
    # Filter to only rows where a route was actually found
    found_df = results_df[results_df["found"] == "Yes"]
    
    if not found_df.empty:
        
        astar_df = found_df[found_df["algorithm"] == "ASTAR"]
        
        if not astar_df.empty:
            best_row = astar_df.iloc[0]
        else:
            best_row = found_df.iloc[0]
        
        path_string = str(best_row["path"])
        route_sites = [int(site.strip()) for site in path_string.split(">")]
        
        print(f"Route loaded: {best_row['algorithm']} | {best_row['model']}")
        print(f"Sites: {route_sites}")
        print(f"Travel time: {best_row['travel_time_minutes']} minutes")
    else:
        print("No routes found in route_results.csv — run a route search in the GUI first.")
else:
    print("route_results.csv not found — run a route search in the GUI first.")

# -----------------------------------------------
# GET TARGET HOUR FROM ROUTE RESULTS
# -----------------------------------------------
TARGET_HOUR = 10
if ROUTE_RESULTS_FILE.exists() and not results_df.empty:
    try:
        departure_time_str = str(results_df.iloc[0]["departure_time"])
        TARGET_HOUR = int(departure_time_str.split(":")[0])
        print(f"Using departure hour from route: {TARGET_HOUR:02d}:00")
    except Exception:
        print("Could not read departure time, using default hour 08:00")

hour_df = hourly_df[hourly_df["hour"] == TARGET_HOUR]
site_avg_traffic = hour_df.groupby("scats_site_number")["traffic_flow"].mean()

# -----------------------------------------------
# COLOR FUNCTION
# -----------------------------------------------
# green = low traffic, orange = moderate, red = high
def get_traffic_color(traffic_value, min_traffic, max_traffic):
    if max_traffic == min_traffic:
        return "green"
    ratio = (traffic_value - min_traffic) / (max_traffic - min_traffic)
    if ratio < 0.33:
        return "green"
    elif ratio < 0.66:
        return "yellow"
    else:
        return "red"

min_traffic = site_avg_traffic.min()
max_traffic = site_avg_traffic.max()

# -----------------------------------------------
# BUILD THE MAP
# -----------------------------------------------
centre_lat = nodes_df["latitude"].mean()
centre_lon = nodes_df["longitude"].mean()
m = folium.Map(location=[centre_lat, centre_lon], zoom_start=14)

# -----------------------------------------------
# DRAW BASE EDGES
# -----------------------------------------------
for _, row in edges_df.iterrows():
    start = int(row["start_scats"])
    end   = int(row["end_scats"])
    start_node = nodes_df[nodes_df["scats_site"] == start]
    end_node   = nodes_df[nodes_df["scats_site"] == end]
    if start_node.empty or end_node.empty:
        continue
    folium.PolyLine(
        locations=[
            [node_coord_lookup[start]["latitude"], node_coord_lookup[start]["longitude"]],
            [node_coord_lookup[end]["latitude"],   node_coord_lookup[end]["longitude"]]
        ],
        color="#3b82f6",
        weight=2,
        opacity=0.5
    ).add_to(m)

# -----------------------------------------------
# DRAW THE ROUTE ON TOP (thick yellow line)
# -----------------------------------------------
if len(route_sites) >= 2:
    for i in range(len(route_sites) - 1):
        start_site = route_sites[i]
        end_site   = route_sites[i + 1]
        
        start_node = nodes_df[nodes_df["scats_site"] == start_site]
        end_node   = nodes_df[nodes_df["scats_site"] == end_site]
        
        if start_node.empty or end_node.empty:
            continue
        
        folium.PolyLine(
            locations=[
                [node_coord_lookup[start_site]["latitude"], node_coord_lookup[start_site]["longitude"]],
                [node_coord_lookup[end_site]["latitude"],   node_coord_lookup[end_site]["longitude"]]
            ],
            color="#f59e0b",  
            weight=6,
            opacity=0.9
        ).add_to(m)

# -----------------------------------------------
# DRAW NODES COLORED BY TRAFFIC
# -----------------------------------------------
origin_site      = route_sites[0] if route_sites else None
destination_site = route_sites[-1] if route_sites else None
route_site_set   = set(route_sites)

for _, row in nodes_df.iterrows():
    site = int(row["scats_site"])
    lat  = node_coord_lookup[site]["latitude"]
    lon  = node_coord_lookup[site]["longitude"]

    if site == origin_site:
        color  = ORIGIN_COLOR
        radius = 12
        traffic_text = f"<br>Avg Traffic at {TARGET_HOUR:02d}:00 — {site_avg_traffic[site]:.0f} vehicles" if site in site_avg_traffic.index else ""
        popup_text = (f"SCATS Site: {site} — ORIGIN{traffic_text}<br>"f"X (Longitude): {lon:.6f}<br>"f"Y (Latitude) : {lat:.6f}")
    elif site == destination_site:
        color  = DESTINATION_COLOR
        radius = 12
        traffic_text = f"<br>Avg Traffic at {TARGET_HOUR:02d}:00 — {site_avg_traffic[site]:.0f} vehicles" if site in site_avg_traffic.index else ""
        popup_text = (f"SCATS Site: {site} — DESTINATION{traffic_text}<br>"f"X (Longitude): {lon:.6f}<br>"f"Y (Latitude) : {lat:.6f}")
    elif site in route_site_set:
        color  = ROUTE_COLOR
        radius = 9
        traffic_text = f"<br>Avg Traffic at {TARGET_HOUR:02d}:00 — {site_avg_traffic[site]:.0f} vehicles" if site in site_avg_traffic.index else ""
        popup_text = (f"SCATS Site: {site} — On route{traffic_text}<br>"f"X (Longitude): {lon:.6f}<br>"f"Y (Latitude) : {lat:.6f}")
    elif site in site_avg_traffic.index:
        traffic_value = site_avg_traffic[site]
        color  = get_traffic_color(traffic_value, min_traffic, max_traffic)
        radius = 7
        popup_text = (f"SCATS Site: {site}<br>"f"Avg Traffic at {TARGET_HOUR:02d}:00 — {traffic_value:.0f} vehicles<br>"f"X (Longitude): {lon:.6f}<br>"f"Y (Latitude) : {lat:.6f}")
    else:
        color  = "gray"
        radius = 7
        popup_text = f"SCATS Site: {site}<br>No data for {TARGET_HOUR:02d}:00"

    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color="white",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_text, max_width=220)
    ).add_to(m)

# -----------------------------------------------
# ADD LEGEND
# -----------------------------------------------
route_info_html = ""
if route_sites:
    route_info_html = f"""
    <br><b>Current Route</b><br>
    <span style="color:#4da3ff;">&#9679;</span> Origin: {origin_site}<br>
    <span style="color:#4cd964;">&#9679;</span> Destination: {destination_site}<br>
    <span style="color:#f59e0b;">&#9679;</span> Route sites & edges<br>
    """

legend_html = f"""
<div style="
    position: fixed;
    bottom: 40px; left: 40px;
    background-color: white;
    padding: 14px 18px;
    border-radius: 8px;
    border: 1px solid #ccc;
    font-family: Arial, sans-serif;
    font-size: 13px;
    z-index: 9999;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
    <b>Traffic at {TARGET_HOUR:02d}:00</b><br>
    <span style="color:green;">&#9679;</span> Low traffic<br>
    <span style="color:yellow;">&#9679;</span> Moderate traffic<br>
    <span style="color:red;">&#9679;</span> High traffic<br>
    <span style="color:gray;">&#9679;</span> No data
    {route_info_html}
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# -----------------------------------------------
# SAVE
# -----------------------------------------------

output_path = Path("scats_map.html")
m.save(output_path)
print(f"Map saved to {output_path.resolve()}")
print(f"Showing traffic at {TARGET_HOUR:02d}:00 with route highlighted.")
print("Open scats_map.html in your browser to view it.")