from pathlib import Path
import json
import re
import pandas as pd
from travel_time import calculate_haversine_distance_km

COORDINATE_FILE = Path("Traffic_Count_Locations_with_LONG_LAT.csv")
PROCESSED_TRAFFIC_FILE = Path("traffic_flow_model_data/processed_hourly_traffic_flow.csv")
SCATS_SITE_LISTING_FILE = Path("SCATSSiteListingSpreadsheet_VicRoads.xlsx")
BOROONDARA_MIN_LAT = -37.95
BOROONDARA_MAX_LAT = -37.75
BOROONDARA_MIN_LON = 144.95
BOROONDARA_MAX_LON = 145.20
OUTPUT_FOLDER = Path("route_graph_data")
OUTPUT_NODES_FILE = OUTPUT_FOLDER / "scats_nodes.csv"
OUTPUT_EDGES_FILE = OUTPUT_FOLDER / "scats_edges_prepared.csv"
OUTPUT_A2A_GRAPH_FILE = OUTPUT_FOLDER / "a2a_base_graph.json"
OUTPUT_A2A_POSITIONS_FILE = OUTPUT_FOLDER / "a2a_node_positions.json"
NEAREST_NEIGHBOURS = 4
MAKE_BIDIRECTIONAL = True

# LOAD DATA
# ============================================================
def load_scats_sites_from_processed_traffic(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Run data-process.py first.")
    df = pd.read_csv(file_path)
    if "scats_site_number" not in df.columns:
        raise ValueError("processed_hourly_traffic_flow.csv must contain scats_site_number.")
    df["scats_site_number"] = pd.to_numeric(df["scats_site_number"], errors="coerce")
    df = df.dropna(subset=["scats_site_number"])
    return sorted(df["scats_site_number"].astype(int).unique())

def clean_column_name(column_name):
    return str(column_name).strip().lower().replace(" ", "_").replace("-", "_")

def read_scats_listing_excel(file_path, sheet_name=None, header=None):
    if file_path.suffix.lower() == ".xlsx":
        return pd.read_excel(file_path,sheet_name=sheet_name,header=header,engine="openpyxl")
    return pd.read_excel(file_path,sheet_name=sheet_name,header=header,engine="xlrd",engine_kwargs={"ignore_workbook_corruption": True})

def load_scats_site_listing(file_path):
    all_sheets = read_scats_listing_excel(file_path, sheet_name=None, header=None)
    
    for sheet_name, raw_df in all_sheets.items():
        for row_index in range(min(40, len(raw_df))):
            row_values = [str(value).strip().lower() for value in raw_df.iloc[row_index].tolist()]
            has_site_number = any("site number" in value for value in row_values)
            has_location_description = any("location description" in value for value in row_values)
            if not (has_site_number and has_location_description):
                continue
            df = read_scats_listing_excel(file_path, sheet_name=sheet_name, header=row_index)
            df.columns = [clean_column_name(column) for column in df.columns]
            site_column = None
            description_column = None
            for column in df.columns:
                if "site" in column and "number" in column:
                    site_column = column
                if "location" in column and "description" in column:
                    description_column = column
            if site_column is None or description_column is None:
                continue
            df = df[[site_column, description_column]].copy()
            df.columns = ["scats_site", "scats_description"]
            df["scats_site"] = pd.to_numeric(df["scats_site"], errors="coerce")
            df = df.dropna(subset=["scats_site", "scats_description"]).copy()
            df["scats_site"] = df["scats_site"].astype(int)
            df["scats_description"] = df["scats_description"].astype(str).str.upper().str.strip()
            return df.drop_duplicates(subset=["scats_site"]).reset_index(drop=True)
    raise ValueError("Could not find SCATS site number and location description columns.")

def normalise_location_text(text):
    text = str(text).upper()
    replacements = {
        "&": " ",
        "/": " ",
        "\\": " ",
        "-": " ",
        "_": " ",
        " ROAD ": " ",
        " RD ": " ",
        " STREET ": " ",
        " ST ": " ",
        " HIGHWAY ": " ",
        " HWY ": " ",
        " AVENUE ": " ",
        " AVE ": " ",
        " DRIVE ": " ",
        " DR ": " ",
        " PARADE ": " ",
        " PDE ": " ",
        " CRESCENT ": " ",
        " CRES ": " "
    }
    text = f" {text} "
    for old_text, new_text in replacements.items():
        text = text.replace(old_text, new_text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_location_tokens(text):
    ignored_words = {"OF", "AT", "AND", "THE", "NEAR", "NR", "NORTH", "SOUTH", "EAST", "WEST", "N", "S", "E", "W", "BD", "BOUND", "BEFORE", "AFTER", "BETWEEN", "BTWN"}
    words = normalise_location_text(text).split()
    return {word for word in words if len(word) >= 3 and word not in ignored_words}

def find_best_coordinate_match(scats_description, coordinate_df):
    scats_tokens = get_location_tokens(scats_description)
    best_score = 0
    best_row = None
    for _, row in coordinate_df.iterrows():
        coordinate_text = f"{row['SITE_DESC']} {row['TFM_DESC']}"
        coordinate_tokens = get_location_tokens(coordinate_text)
        if not scats_tokens:
            continue
        matched_tokens = scats_tokens.intersection(coordinate_tokens)
        score = len(matched_tokens) / len(scats_tokens)
        if score > best_score:
            best_score = score
            best_row = row
    if best_score < 0.6:
        return None
    return best_row

def load_scats_coordinates(coordinate_file, scats_listing_df, required_scats_sites):
    if not coordinate_file.exists():
        raise FileNotFoundError(f"{coordinate_file} not found.")
    coordinate_df = pd.read_csv(coordinate_file)
    required_columns = {"X", "Y", "SITE_DESC", "TFM_DESC"}
    if not required_columns.issubset(coordinate_df.columns):
        raise ValueError("Traffic_Count_Locations_with_LONG_LAT.csv must contain X, Y, SITE_DESC, and TFM_DESC columns.")
    coordinate_df = coordinate_df.copy()
    coordinate_df["longitude"] = pd.to_numeric(coordinate_df["X"], errors="coerce")
    coordinate_df["latitude"] = pd.to_numeric(coordinate_df["Y"], errors="coerce")
    coordinate_df = coordinate_df.dropna(subset=["latitude", "longitude"]).copy()

    # Keep only likely Boroondara-area coordinates.
    coordinate_df = coordinate_df[(coordinate_df["latitude"].between(BOROONDARA_MIN_LAT, BOROONDARA_MAX_LAT)) &(coordinate_df["longitude"].between(BOROONDARA_MIN_LON, BOROONDARA_MAX_LON))].copy()

    if coordinate_df.empty:
        raise ValueError("No coordinate rows found inside the Boroondara coordinate range.")
    scats_listing_df = scats_listing_df[scats_listing_df["scats_site"].isin(required_scats_sites)].copy()
    matched_rows = []
    unmatched_sites = []

    for _, scats_row in scats_listing_df.iterrows():
        scats_site = int(scats_row["scats_site"])
        scats_description = scats_row["scats_description"]
        best_match = find_best_coordinate_match(scats_description, coordinate_df)
        if best_match is None:
            unmatched_sites.append(scats_site)
            continue
        matched_rows.append({
            "scats_site": scats_site,
            "location_description": scats_description,
            "matched_coordinate_description": best_match["SITE_DESC"],
            "latitude": float(best_match["latitude"]),
            "longitude": float(best_match["longitude"])
        })

    if unmatched_sites:
        print("Warning: some SCATS sites could not be matched to coordinates:")
        print(unmatched_sites)
    if not matched_rows:
        raise ValueError("No SCATS sites could be matched to coordinates.")
    nodes_df = pd.DataFrame(matched_rows)
    nodes_df = (nodes_df.drop_duplicates(subset=["scats_site"]).sort_values("scats_site").reset_index(drop=True))
    return nodes_df

# BUILD APPROXIMATE SCATS GRAPH
# ============================================================
def build_nearest_neighbour_edges(nodes_df):
    edges = []
    for _, start_row in nodes_df.iterrows():
        start_site = int(start_row["scats_site"])
        start_lat = float(start_row["latitude"])
        start_lon = float(start_row["longitude"])
        distances = []
        for _, end_row in nodes_df.iterrows():
            end_site = int(end_row["scats_site"])
            if start_site == end_site:
                continue
            distance_km = calculate_haversine_distance_km(start_lat,start_lon,float(end_row["latitude"]),float(end_row["longitude"]))

            # Skip duplicate/same-location matches.
            if distance_km <= 0.01:
                continue

            distances.append({
                "start_scats": start_site,
                "end_scats": end_site,
                "distance_km": round(distance_km, 4)
            })
        nearest_edges = sorted(distances, key=lambda item: item["distance_km"])[:NEAREST_NEIGHBOURS]
        edges.extend(nearest_edges)
    edges_df = pd.DataFrame(edges)
    if MAKE_BIDIRECTIONAL:
        reverse_edges_df = edges_df.rename(
            columns={
                "start_scats": "end_scats",
                "end_scats": "start_scats"
            }
        )
        edges_df = pd.concat([edges_df, reverse_edges_df], ignore_index=True)
    edges_df = (edges_df.sort_values(["start_scats", "distance_km", "end_scats"]).drop_duplicates(subset=["start_scats", "end_scats"]).reset_index(drop=True))
    return edges_df

def build_a2a_base_graph(edges_df):
    graph = {}
    for _, row in edges_df.iterrows():
        start = int(row["start_scats"])
        end = int(row["end_scats"])
        distance = float(row["distance_km"])
        graph.setdefault(str(start), [])
        graph.setdefault(str(end), [])
        graph[str(start)].append([end, distance])
    return graph

def build_a2a_node_positions(nodes_df):
    node_positions = {}
    for _, row in nodes_df.iterrows():
        scats_site = int(row["scats_site"])
        node_positions[str(scats_site)] = [float(row["longitude"]),float(row["latitude"])]
    return node_positions

# SAVE OUTPUTS
# ============================================================
def save_outputs(nodes_df, edges_df, a2a_graph, a2a_node_positions):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(OUTPUT_NODES_FILE, index=False)
    edges_df.to_csv(OUTPUT_EDGES_FILE, index=False)
    with open(OUTPUT_A2A_GRAPH_FILE, "w", encoding="utf-8") as file:
        json.dump(a2a_graph, file, indent=4)
    with open(OUTPUT_A2A_POSITIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(a2a_node_positions, file, indent=4)

# MAIN
# ============================================================
def main():
    print("Loading SCATS sites from processed traffic data...")
    required_scats_sites = load_scats_sites_from_processed_traffic(PROCESSED_TRAFFIC_FILE)
    print("Loading SCATS site listing...")
    scats_listing_df = load_scats_site_listing(SCATS_SITE_LISTING_FILE)
    print("Loading and matching SCATS coordinates...")
    nodes_df = load_scats_coordinates(COORDINATE_FILE,scats_listing_df,required_scats_sites)
    print("Building approximate SCATS graph from nearest neighbours...")
    edges_df = build_nearest_neighbour_edges(nodes_df)
    print("Building A2A-compatible graph files...")
    a2a_graph = build_a2a_base_graph(edges_df)
    a2a_node_positions = build_a2a_node_positions(nodes_df)
    print("Saving graph files...")
    save_outputs(nodes_df, edges_df, a2a_graph, a2a_node_positions)
    
    print()
    print("SCATS graph preparation complete.")
    print(f"SCATS nodes saved to: {OUTPUT_NODES_FILE}")
    print(f"SCATS edges saved to: {OUTPUT_EDGES_FILE}")
    print(f"A2A graph saved to: {OUTPUT_A2A_GRAPH_FILE}")
    print(f"A2A node positions saved to: {OUTPUT_A2A_POSITIONS_FILE}")
    print()
    print(f"Nodes created: {len(nodes_df)}")
    print(f"Edges created: {len(edges_df)}")

if __name__ == "__main__":
    main()