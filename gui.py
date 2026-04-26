from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import pandas as pd
from route_engine import find_routes
import random

ROUTE_GRAPH_FOLDER = Path("route_graph_data")
SCATS_NODES_FILE = ROUTE_GRAPH_FOLDER / "scats_nodes.csv"
SCATS_EDGES_FILE = ROUTE_GRAPH_FOLDER / "scats_edges_prepared.csv"
DEFAULT_ORIGIN = 2000
DEFAULT_DESTINATION = 3002
DEFAULT_DATETIME = "2006-10-15 15:00:00"
BASE_EDGE_COLOR = "#384050"
NODE_COLOR = "#d8dee9"
ORIGIN_COLOR = "#4da3ff"
DESTINATION_COLOR = "#4cd964"
ROUTE_COLOR = "#ffb000"
BACKGROUND_COLOR = "#111827"

class TBRGSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Traffic-Based Route Guidance System")
        self.geometry("1450x850")
        self.minsize(1200, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.nodes_df = self.load_nodes()
        self.edges_df = self.load_edges()
        self.node_lookup = self.build_node_lookup()
        self.site_options = self.build_site_options()
        self.site_display_lookup = self.build_site_display_lookup()
        self.route_paths = {}
        self.current_path = []
        self.latest_results_by_model = {}
        self.current_route_model = None
        self.edge_cost_mode = ctk.StringVar(value="route")
        self.node_xy_cache = {}
        self.map_zoom = 1.0
        self.map_pan_x = 0
        self.map_pan_y = 0
        self.last_pan_x = None
        self.last_pan_y = None
        self.create_layout()
        self.draw_map()
        
    # ============================================================
    # DATA LOADING
    # ============================================================
    def load_nodes(self):
        if not SCATS_NODES_FILE.exists():
            raise FileNotFoundError("route_graph_data/scats_nodes.csv not found. Run prepare_scats_graph.py first.")
        nodes_df = pd.read_csv(SCATS_NODES_FILE)
        required_columns = {"scats_site", "latitude", "longitude"}
        if not required_columns.issubset(nodes_df.columns):
            raise ValueError("scats_nodes.csv must contain scats_site, latitude, and longitude.")
        nodes_df["scats_site"] = nodes_df["scats_site"].astype(int)
        return nodes_df.sort_values("scats_site").reset_index(drop=True)

    def load_edges(self):
        if not SCATS_EDGES_FILE.exists():
            raise FileNotFoundError("route_graph_data/scats_edges_prepared.csv not found. Run prepare_scats_graph.py first.")
        edges_df = pd.read_csv(SCATS_EDGES_FILE)
        required_columns = {"start_scats", "end_scats", "distance_km"}
        if not required_columns.issubset(edges_df.columns):
            raise ValueError("scats_edges_prepared.csv must contain start_scats, end_scats, and distance_km.")
        edges_df["start_scats"] = edges_df["start_scats"].astype(int)
        edges_df["end_scats"] = edges_df["end_scats"].astype(int)
        return edges_df

    def build_node_lookup(self):
        lookup = {}
        for _, row in self.nodes_df.iterrows():
            site = int(row["scats_site"])
            
            # Add a microscopic random jitter to push nodes a few meters apart in the logical map (roughly 40 meters)
            lat_jitter = random.uniform(-0.0004, 0.0004)
            lon_jitter = random.uniform(-0.0004, 0.0004)
            
            lookup[site] = {
                "latitude": float(row["latitude"]) + lat_jitter,
                "longitude": float(row["longitude"]) + lon_jitter,
                "location": str(row.get("location_description", "Unknown location"))
            }
        return lookup

    def build_site_options(self):
        options = []
        for _, row in self.nodes_df.iterrows():
            site = int(row["scats_site"])
            location = str(row.get("location_description", "Unknown location"))
            short_location = location[:55]
            options.append(f"{site} - {short_location}")
        return options

    def build_site_display_lookup(self):
        lookup = {}
        for option in self.site_options:
            site = int(option.split(" - ")[0])
            lookup[site] = option
        return lookup

    # GUI LAYOUT
    # ==========
    def create_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=360, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_tabs()
        
    def create_main_tabs(self):
        self.main_tabs = ctk.CTkTabview(self.main_frame)
        self.main_tabs.grid(row=0, column=0, sticky="nsew")
        self.map_tab = self.main_tabs.add("Graph Map")
        self.tables_tab = self.main_tabs.add("Results Tables")

        self.map_tab.grid_rowconfigure(0, weight=1)
        self.map_tab.grid_columnconfigure(0, weight=1)
        self.tables_tab.grid_rowconfigure(0, weight=1)
        self.tables_tab.grid_columnconfigure(0, weight=1)

        self.create_map_panel(self.map_tab)
        self.create_results_panel(self.tables_tab)

    def create_sidebar(self):
        title = ctk.CTkLabel(self.sidebar,text="TBRGS Control Panel",font=ctk.CTkFont(size=22, weight="bold"))
        title.pack(anchor="w", padx=20, pady=(20, 12))
        subtitle = ctk.CTkLabel(self.sidebar,text="Select route inputs and run pathfinding.",font=ctk.CTkFont(size=13),text_color="#aab2c0",wraplength=310,justify="left")
        subtitle.pack(anchor="w", padx=20, pady=(0, 18))

        self.origin_var = ctk.StringVar(value=self.get_default_site_display(DEFAULT_ORIGIN))
        self.destination_var = ctk.StringVar(value=self.get_default_site_display(DEFAULT_DESTINATION))
        self.datetime_var = ctk.StringVar(value=DEFAULT_DATETIME)
        self.model_var = ctk.StringVar(value="GRU")
        self.top_k_var = ctk.StringVar(value="5")
        self.click_target_var = ctk.StringVar(value="Origin")

        self.add_sidebar_label("Origin SCATS")
        self.origin_combo = ctk.CTkComboBox(self.sidebar,values=self.site_options,variable=self.origin_var,width=310, state="readonly")
        self.origin_combo.pack(anchor="w", padx=20, pady=(0, 10))

        self.add_sidebar_label("Destination SCATS")
        self.destination_combo = ctk.CTkComboBox(self.sidebar,values=self.site_options,variable=self.destination_var,width=310, state="readonly")
        self.destination_combo.pack(anchor="w", padx=20, pady=(0, 10))

        self.add_sidebar_label("Departure datetime")
        self.datetime_entry = ctk.CTkEntry(self.sidebar,textvariable=self.datetime_var,width=310,placeholder_text="YYYY-MM-DD HH:MM:SS")
        self.datetime_entry.pack(anchor="w", padx=20, pady=(0, 10))

        self.add_sidebar_label("Model mode")
        self.model_combo = ctk.CTkComboBox(self.sidebar,values=["GRU", "LSTM", "Compare GRU + LSTM"],variable=self.model_var,width=310, state="readonly")
        self.model_combo.pack(anchor="w", padx=20, pady=(0, 10))

        self.add_sidebar_label("Top-k route alternatives")
        self.top_k_combo = ctk.CTkComboBox(self.sidebar,values=["1", "2", "3", "4", "5"],variable=self.top_k_var,width=310, state="readonly")
        self.top_k_combo.pack(anchor="w", padx=20, pady=(0, 16))

        self.add_sidebar_label("Algorithms to run")
        self.algorithm_vars = {}

        algorithm_frame = ctk.CTkFrame(self.sidebar)
        algorithm_frame.pack(anchor="w", padx=20, pady=(0, 14), fill="x")
        algorithms = [
            ("bfs", "BFS"),
            ("dfs", "DFS"),
            ("gbfs", "GBFS"),
            ("astar", "A*"),
            ("cus1", "CUS1"),
            ("cus2", "CUS2")
        ]

        for index, (value, label) in enumerate(algorithms):
            var = ctk.BooleanVar(value=True)
            self.algorithm_vars[value] = var
            checkbox = ctk.CTkCheckBox(algorithm_frame,text=label,variable=var)
            checkbox.grid(row=index // 2, column=index % 2, sticky="w", padx=12, pady=8)

        self.add_sidebar_label("Toggle left-click mode")
        self.click_mode = ctk.CTkSegmentedButton(self.sidebar,values=["Origin", "Destination"],variable=self.click_target_var,width=310)
        self.click_mode.pack(anchor="w", padx=20, pady=(0, 16))
        self.find_button = ctk.CTkButton(self.sidebar,text="Find Routes",command=self.on_find_routes,height=42,font=ctk.CTkFont(size=15, weight="bold"))
        self.find_button.pack(anchor="w", padx=20, pady=(0, 14), fill="x")
        self.reset_button = ctk.CTkButton(self.sidebar,text="Reset",command=self.reset_route_results,height=38,fg_color="#b91c1c",hover_color="#991b1b",font=ctk.CTkFont(size=14, weight="bold"))
        self.reset_button.pack(anchor="w", padx=20, pady=(0, 14), fill="x")

        self.add_sidebar_label("Highlighted route")
        self.selected_route_var = ctk.StringVar(value="No route yet")
        self.route_combo = ctk.CTkComboBox(self.sidebar,values=["No route yet"],variable=self.selected_route_var,command=self.on_route_selected,width=310, state="readonly")
        self.route_combo.pack(anchor="w", padx=20, pady=(0, 14))
        
        self.add_sidebar_label("Edge cost visibility")
        edge_cost_frame = ctk.CTkFrame(self.sidebar)
        edge_cost_frame.pack(anchor="w", padx=20, pady=(0, 14), fill="x")
        self.show_all_costs_var = ctk.BooleanVar(value=False)
        self.show_route_costs_var = ctk.BooleanVar(value=True)
        self.hide_costs_var = ctk.BooleanVar(value=False)

        self.show_all_costs_check = ctk.CTkCheckBox(edge_cost_frame,text="Show all edge costs",variable=self.show_all_costs_var,command=lambda: self.set_edge_cost_mode("all"))
        self.show_all_costs_check.pack(anchor="w", padx=12, pady=(8, 4))
        self.show_route_costs_check = ctk.CTkCheckBox(edge_cost_frame,text="Show highlighted route costs",variable=self.show_route_costs_var,command=lambda: self.set_edge_cost_mode("route"))
        self.show_route_costs_check.pack(anchor="w", padx=12, pady=4)
        self.hide_costs_check = ctk.CTkCheckBox(edge_cost_frame,text="Hide edge costs",variable=self.hide_costs_var,command=lambda: self.set_edge_cost_mode("hide"))
        self.hide_costs_check.pack(anchor="w", padx=12, pady=(4, 8))

        self.status_label = ctk.CTkLabel(self.sidebar,text="Ready.",text_color="#aab2c0",wraplength=310,justify="left")
        self.status_label.pack(anchor="w", padx=20, pady=(8, 0))

    def add_sidebar_label(self, text):
        label = ctk.CTkLabel(self.sidebar,text=text,font=ctk.CTkFont(size=13, weight="bold"))
        label.pack(anchor="w", padx=20, pady=(4, 4))

    def create_map_panel(self, parent):
        self.map_frame = ctk.CTkFrame(parent)
        self.map_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.map_frame.grid_rowconfigure(2, weight=1) 
        self.map_frame.grid_columnconfigure(0, weight=1)

        map_title = ctk.CTkLabel(self.map_frame, text="Boroondara SCATS Graph Map", font=ctk.CTkFont(size=20, weight="bold"))
        map_title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 0))

        instruction_text = "Left-click node: set as Origin/Destination   |   Mouse wheel: zoom   |   Right-click drag: pan   |   Double right-click: reset view"
        map_instructions = ctk.CTkLabel(self.map_frame, text=instruction_text, font=ctk.CTkFont(size=13), text_color="#9ca3af")
        map_instructions.grid(row=1, column=0, sticky="w", padx=14, pady=(2, 8))

        self.canvas = tk.Canvas(self.map_frame, background=BACKGROUND_COLOR, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.canvas.bind("<Configure>", lambda event: self.draw_map())
        self.canvas.bind("<Button-1>", self.on_map_click)

        # Mouse wheel zooms the drawn graph around the cursor.
        self.canvas.bind("<MouseWheel>", self.on_map_zoom)

        # Right-click drag moves the map without interfering with left-click node selection.
        self.canvas.bind("<ButtonPress-3>", self.start_map_pan)
        self.canvas.bind("<B3-Motion>", self.on_map_pan)
        self.canvas.bind("<ButtonRelease-3>", self.end_map_pan)

        # Double right-click resets the map view.
        self.canvas.bind("<Double-Button-3>", self.reset_map_view)

    def create_results_panel(self, parent):
        self.results_panel = ctk.CTkTabview(parent)
        self.results_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.algorithm_tab = self.results_panel.add("Algorithm comparison")
        self.top_k_tab = self.results_panel.add("Top-k routes")
        self.configure_treeview_style()
        self.algorithm_tree = self.create_treeview(
            parent=self.algorithm_tab,
            columns=("model", "algorithm", "found", "time", "nodes", "path"),
            headings={
                "model": "Model",
                "algorithm": "Algorithm",
                "found": "Found",
                "time": "Travel time",
                "nodes": "Nodes created",
                "path": "Path"
            },
            widths={
                "model": 80,
                "algorithm": 90,
                "found": 70,
                "time": 110,
                "nodes": 110,
                "path": 650
            }
        )

        self.top_k_tree = self.create_treeview(
            parent=self.top_k_tab,
            columns=("model", "route", "time", "distance", "path"),
            headings={
                "model": "Model",
                "route": "Route",
                "time": "Travel time",
                "distance": "Distance",
                "path": "Path"
            },
            widths={
                "model": 80,
                "route": 70,
                "time": 110,
                "distance": 100,
                "path": 750
            }
        )

    def configure_treeview_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",background="#1f2937",foreground="#f9fafb",fieldbackground="#1f2937",rowheight=28,borderwidth=0)
        style.configure("Treeview.Heading",background="#374151",foreground="#f9fafb",relief="flat",font=("Segoe UI", 10, "bold"))
        style.map("Treeview",background=[("selected", "#2563eb")],foreground=[("selected", "#ffffff")])

    def create_treeview(self, parent, columns, headings, widths):
        frame = tk.Frame(parent, bg="#1f2937")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        tree = ttk.Treeview(frame,columns=columns,show="headings")
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return tree

    # MAP DRAWING
    # ===========
    def draw_map(self):
        self.canvas.delete("all")
        self.node_xy_cache = {}
        width = max(self.canvas.winfo_width(), 600)
        height = max(self.canvas.winfo_height(), 400)
        if width <= 1 or height <= 1:
            return
        self.draw_base_edges(width, height)
        self.draw_current_route(width, height)
        self.draw_edge_costs(width, height)
        self.draw_nodes(width, height)
        self.draw_map_legend(width, height)

    def draw_base_edges(self, width, height):
        drawn_edges = set()
        for _, row in self.edges_df.iterrows():
            start = int(row["start_scats"])
            end = int(row["end_scats"])
            if start not in self.node_lookup or end not in self.node_lookup:
                continue

            # Draw A-B only once even if the data also contains B-A.
            edge_key = tuple(sorted((start, end)))

            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            x1, y1 = self.latlon_to_canvas(self.node_lookup[start]["latitude"],self.node_lookup[start]["longitude"],width,height)
            x2, y2 = self.latlon_to_canvas(self.node_lookup[end]["latitude"],self.node_lookup[end]["longitude"],width,height)
            self.canvas.create_line(x1,y1,x2,y2,fill=BASE_EDGE_COLOR,width=1)

    def draw_current_route(self, width, height):
        if len(self.current_path) < 2:
            return
        for start, end in zip(self.current_path[:-1], self.current_path[1:]):
            if start not in self.node_lookup or end not in self.node_lookup:
                continue
            x1, y1 = self.latlon_to_canvas(self.node_lookup[start]["latitude"],self.node_lookup[start]["longitude"],width,height)
            x2, y2 = self.latlon_to_canvas(self.node_lookup[end]["latitude"],self.node_lookup[end]["longitude"],width,height)
            visual_scale = max(0.8, min(1.4, self.map_zoom))
            self.canvas.create_line(x1, y1, x2, y2, fill=ROUTE_COLOR, width=5 * visual_scale, arrow=tk.LAST, arrowshape=(int(14 * visual_scale),int(16 * visual_scale),int(6 * visual_scale)))

    def draw_nodes(self, width, height):
        origin_site = self.parse_site_selection(self.origin_var.get())
        destination_site = self.parse_site_selection(self.destination_var.get())
        route_sites = set(self.current_path)

        # Keep visual items readable without making them grow too aggressively when zoomed in.
        visual_scale = max(0.9, min(1.25, self.map_zoom ** 0.25))

        for site, data in self.node_lookup.items():
            x, y = self.latlon_to_canvas(data["latitude"],data["longitude"],width,height)
            self.node_xy_cache[site] = (x, y)
            radius = 5 * visual_scale
            fill_color = NODE_COLOR
            if site in route_sites:
                radius = 7 * visual_scale
                fill_color = ROUTE_COLOR
            if site == origin_site:
                radius = 8 * visual_scale
                fill_color = ORIGIN_COLOR
            if site == destination_site:
                radius = 8 * visual_scale
                fill_color = DESTINATION_COLOR
            self.canvas.create_oval(x - radius,y - radius,x + radius,y + radius,fill=fill_color,outline="#111827",width=1)
            label_font_size = int(8 * visual_scale)

            # Label gap is based on node radius, so the number stays separated from the node.
            label_gap = radius + 8

            label_x = x + label_gap
            label_y = y - label_gap
            label_anchor = "w"

            # If the label is near the right side, place it on the left instead.
            if x > width - 90:
                label_x = x - label_gap
                label_anchor = "e"

            # If the label is near the top, place it below instead.
            if y < 35:
                label_y = y + label_gap

            self.canvas.create_text(label_x,label_y,text=str(site),fill="#e5e7eb",font=("Segoe UI", label_font_size),anchor=label_anchor)

    def draw_map_legend(self, width, height):
        x = 18
        y = height - 92
        self.canvas.create_rectangle(x,y,x + 210,y + 74,fill="#1f2937",outline="#4b5563")
        self.canvas.create_text(x + 12,y + 12,text="Legend",fill="#f9fafb",font=("Segoe UI", 10, "bold"),anchor="w")
        self.draw_legend_item(x + 14, y + 34, ORIGIN_COLOR, "Origin")
        self.draw_legend_item(x + 95, y + 34, DESTINATION_COLOR, "Destination")
        self.draw_legend_item(x + 14, y + 56, ROUTE_COLOR, "Highlighted route")

    def draw_legend_item(self, x, y, color, text):
        self.canvas.create_oval(x, y - 5, x + 10, y + 5, fill=color, outline="")
        self.canvas.create_text(x + 16,y,text=text,fill="#d1d5db",font=("Segoe UI", 9),anchor="w")
    
    def get_edge_cost_lookup(self):
        if not self.latest_results_by_model:
            return {}
        if self.current_route_model is None:
            model_type = next(iter(self.latest_results_by_model))
        else:
            model_type = self.current_route_model
        if model_type not in self.latest_results_by_model:
            return {}
        edges_with_time_df = self.latest_results_by_model[model_type]["edges_with_time"]
        cost_lookup = {}
        for _, row in edges_with_time_df.iterrows():
            start = int(row["start_scats"])
            end = int(row["end_scats"])
            cost_lookup[(start, end)] = float(row["travel_time_minutes"])
        return cost_lookup


    def draw_edge_cost_label(self, x1, y1, x2, y2, cost):
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)

        # Offset label slightly away from the edge so it does not sit directly on the line.
        label_x = mid_x + (-dy / length) * 10
        label_y = mid_y + (dx / length) * 10
        
        text = f"{cost:.2f}"
        self.canvas.create_rectangle(label_x - 18,label_y - 9,label_x + 18,label_y + 9,fill="#1f2937",outline="#4b5563")
        self.canvas.create_text(label_x,label_y,text=text,fill="#f9fafb",font=("Segoe UI", 8, "bold"))

    def draw_edge_costs(self, width, height):
        mode = self.edge_cost_mode.get()
        if mode == "hide":
            return
        cost_lookup = self.get_edge_cost_lookup()
        if not cost_lookup:
            return
        if mode == "route":
            edge_pairs = list(zip(self.current_path[:-1], self.current_path[1:]))
        else:
            edge_pairs = list(cost_lookup.keys())
        for start, end in edge_pairs:
            if start not in self.node_lookup or end not in self.node_lookup:
                continue
            cost = cost_lookup.get((start, end))
            if cost is None:
                continue
            x1, y1 = self.latlon_to_canvas(self.node_lookup[start]["latitude"],self.node_lookup[start]["longitude"],width,height)
            x2, y2 = self.latlon_to_canvas(self.node_lookup[end]["latitude"],self.node_lookup[end]["longitude"],width,height)
            self.draw_edge_cost_label(x1, y1, x2, y2, cost)

    def latlon_to_canvas(self, latitude, longitude, width, height):
        padding = 55
        min_lat = self.nodes_df["latitude"].min()
        max_lat = self.nodes_df["latitude"].max()
        min_lon = self.nodes_df["longitude"].min()
        max_lon = self.nodes_df["longitude"].max()
        lon_range = max(max_lon - min_lon, 0.000001)
        lat_range = max(max_lat - min_lat, 0.000001)
        base_x = padding + ((longitude - min_lon) / lon_range) * (width - 2 * padding)

        # Latitude increases upward geographically, but canvas y increases downward.
        base_y = height - padding - ((latitude - min_lat) / lat_range) * (height - 2 * padding)

        # Zoom is applied from the top-left origin, then pan shifts the whole map.
        x = base_x * self.map_zoom + self.map_pan_x
        y = base_y * self.map_zoom + self.map_pan_y

        return x, y

    # USER ACTIONS
    # ============
    def on_map_click(self, event):
        nearest_site = self.get_nearest_site(event.x, event.y)
        if nearest_site is None:
            return
        display_value = self.site_display_lookup[nearest_site]
        if self.click_target_var.get() == "Origin":
            self.origin_var.set(display_value)
        else:
            self.destination_var.set(display_value)
        self.draw_map()
    
    def on_map_zoom(self, event):
        old_zoom = self.map_zoom
        if event.delta > 0:
            zoom_multiplier = 1.15
        else:
            zoom_multiplier = 1 / 1.15
            
        # Max zoom limit
        new_zoom = max(0.5, min(30.0, old_zoom * zoom_multiplier))
        if new_zoom == old_zoom:
            return
        mouse_x = event.x
        mouse_y = event.y

        # Convert the mouse position back into the unzoomed map coordinate.
        map_x_under_mouse = (mouse_x - self.map_pan_x) / old_zoom
        map_y_under_mouse = (mouse_y - self.map_pan_y) / old_zoom

        self.map_zoom = new_zoom

        # Recalculate pan so the same map point stays under the mouse cursor.
        self.map_pan_x = mouse_x - (map_x_under_mouse * new_zoom)
        self.map_pan_y = mouse_y - (map_y_under_mouse * new_zoom)

        self.draw_map()

    def start_map_pan(self, event):
        self.last_pan_x = event.x
        self.last_pan_y = event.y

    def on_map_pan(self, event):
        if self.last_pan_x is None or self.last_pan_y is None:
            return
        dx = event.x - self.last_pan_x
        dy = event.y - self.last_pan_y
        self.map_pan_x += dx
        self.map_pan_y += dy
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.draw_map()

    def end_map_pan(self, event):
        self.last_pan_x = None
        self.last_pan_y = None

    def reset_map_view(self, event=None):
        self.map_zoom = 1.0
        self.map_pan_x = 0
        self.map_pan_y = 0
        self.draw_map()

    def get_nearest_site(self, click_x, click_y):
        if not self.node_xy_cache:
            return None
        nearest_site = None
        nearest_distance = float("inf")
        for site, (node_x, node_y) in self.node_xy_cache.items():
            distance = ((click_x - node_x) ** 2 + (click_y - node_y) ** 2) ** 0.5
            if distance < nearest_distance:
                nearest_site = site
                nearest_distance = distance
        if nearest_distance <= 18:
            return nearest_site
        return None

    def on_find_routes(self):
        try:
            origin = self.parse_site_selection(self.origin_var.get())
            destination = self.parse_site_selection(self.destination_var.get())
            
            # Validation: Prevent same origin and destination
            if origin == destination:
                messagebox.showerror("Invalid input", "Origin and Destination SCATS cannot be the same.")
                return

            # Validation: Catch bad datetime formats gracefully
            departure_datetime = self.datetime_var.get().strip()
            try:
                pd.to_datetime(departure_datetime)
            except Exception:
                messagebox.showerror("Invalid input", "Invalid datetime format.\n\nPlease use: YYYY-MM-DD HH:MM:SS\nExample: 2006-10-15 15:00:00")
                return

            top_k = int(self.top_k_var.get())
            selected_algorithms = [algorithm_name for algorithm_name, variable in self.algorithm_vars.items() if variable.get()]
            
            # Validation: Ensure at least one algorithm is checked
            if not selected_algorithms:
                messagebox.showerror("Invalid input", "Please select at least one search algorithm to run.")
                return
                
            model_mode = self.model_var.get()
            if model_mode == "Compare GRU + LSTM":
                model_types = ["gru", "lstm"]
            else:
                model_types = [model_mode.lower()]
        except Exception as error:
            messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(error)}")
            return
        self.find_button.configure(state="disabled", text="Running...")
        self.status_label.configure(text="Predicting traffic flow and running search algorithms...")

        # Route calculation may take a few seconds, so it runs in a worker thread.
        worker = threading.Thread(target=self.run_route_search_worker,args=(origin, destination, departure_datetime, model_types, top_k, selected_algorithms),daemon=True)
        worker.start()

    def run_route_search_worker(self, origin, destination, departure_datetime, model_types, top_k, selected_algorithms):
        try:
            results_by_model = {}
            for model_type in model_types:
                results_by_model[model_type] = find_routes(origin_scats=origin,destination_scats=destination,departure_datetime=departure_datetime,model_type=model_type,top_k_routes=top_k,algorithm_names=selected_algorithms)
            self.after(0, lambda: self.on_route_search_finished(results_by_model))
        except Exception as error:
            self.after(0, lambda: self.on_route_search_failed(str(error)))

    def on_route_search_finished(self, results_by_model):
        self.find_button.configure(state="normal", text="Find Routes")
        self.status_label.configure(text="Route search complete.")
        self.latest_results_by_model = results_by_model
        self.populate_algorithm_results(results_by_model)
        self.populate_top_k_results(results_by_model)
        self.update_route_selector(results_by_model)

    def on_route_search_failed(self, error_message):
        self.find_button.configure(state="normal", text="Find Routes")
        self.status_label.configure(text="Route search failed.")
        messagebox.showerror("Route search failed", error_message)

    def on_route_selected(self, selected_route):
        if selected_route not in self.route_paths:
            return
        route_info = self.route_paths[selected_route]
        self.current_path = route_info["path"]
        self.current_route_model = route_info["model_type"]
        self.draw_map()
    
    def set_edge_cost_mode(self, mode):
        self.edge_cost_mode.set(mode)
        self.show_all_costs_var.set(mode == "all")
        self.show_route_costs_var.set(mode == "route")
        self.hide_costs_var.set(mode == "hide")
        self.draw_map()
    
    def reset_route_results(self):
        # Clear route-search outputs only. Keep user-selected inputs unchanged.
        self.route_paths = {}
        self.current_path = []
        self.latest_results_by_model = {}
        self.current_route_model = None
        self.selected_route_var.set("No route yet")
        self.route_combo.configure(values=["No route yet"])
        self.algorithm_tree.delete(*self.algorithm_tree.get_children())
        self.top_k_tree.delete(*self.top_k_tree.get_children())
        self.set_edge_cost_mode("route")
        self.status_label.configure(text="Ready.")
        self.draw_map()

    # RESULT DISPLAY
    # ==============
    def populate_algorithm_results(self, results_by_model):
        self.algorithm_tree.delete(*self.algorithm_tree.get_children())

        for model_type, result in results_by_model.items():
            algorithm_results = result["algorithm_results"]
            for algorithm_name, algorithm_result in algorithm_results.items():
                found = "Yes" if algorithm_result["found"] else "No"
                travel_time = self.format_minutes(algorithm_result["travel_time_minutes"])
                nodes_created = str(algorithm_result["nodes_created"])
                path_text = self.format_path(algorithm_result["path"])
                self.algorithm_tree.insert("","end",values=(model_type.upper(),algorithm_name.upper(),found,travel_time,nodes_created,path_text))

    def populate_top_k_results(self, results_by_model):
        self.top_k_tree.delete(*self.top_k_tree.get_children())

        for model_type, result in results_by_model.items():
            top_k_routes = result["top_k_routes"]
            for route in top_k_routes:
                self.top_k_tree.insert("", "end", values=( model_type.upper(), route["route_number"], self.format_minutes(route["travel_time_minutes"]), f"{route['distance_km']:.2f} km", self.format_path(route["path"])))

    def update_route_selector(self, results_by_model):
        self.route_paths = {}

        for model_type, result in results_by_model.items():
            for algorithm_name, algorithm_result in result["algorithm_results"].items():
                if not algorithm_result["found"]:
                    continue
                label = (f"{model_type.upper()} | " f"{algorithm_name.upper()} | " f"{algorithm_result['travel_time_minutes']:.2f} min")
                self.route_paths[label] = {
                    "path": algorithm_result["path"],
                    "model_type": model_type
                }
            for route in result["top_k_routes"]:
                label = (f"{model_type.upper()} | " f"TOP-K {route['route_number']} | " f"{route['travel_time_minutes']:.2f} min")
                self.route_paths[label] = {
                    "path": route["path"],
                    "model_type": model_type
                }
        if not self.route_paths:
            self.route_combo.configure(values=["No route available"])
            self.selected_route_var.set("No route available")
            self.current_path = []
            self.draw_map()
            return
        
        route_options = list(self.route_paths.keys())
        self.route_combo.configure(values=route_options)
        default_route = self.choose_default_route(route_options)
        self.selected_route_var.set(default_route)
        route_info = self.route_paths[default_route]
        self.current_path = route_info["path"]
        self.current_route_model = route_info["model_type"]
        self.draw_map()

    def choose_default_route(self, route_options):
        for option in route_options:
            if "| ASTAR |" in option:
                return option
        return route_options[0]

    # HELPERS
    # =======
    def get_default_site_display(self, site):
        if site in self.site_display_lookup:
            return self.site_display_lookup[site]
        return self.site_options[0]

    def parse_site_selection(self, selection_text):
        return int(str(selection_text).split(" - ")[0])

    def format_path(self, path):
        if not path:
            return "-"
        return " → ".join(str(site) for site in path)

    def format_minutes(self, value):
        if value is None:
            return "-"
        return f"{float(value):.2f} min"

if __name__ == "__main__":
    app = TBRGSApp()
    app.mainloop()