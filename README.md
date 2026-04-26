# COS30019-IntroductionToAI-Assignment2B
Traffic-Based Route Guidance System (TBRGS) for COS30019 Introduction to Artificial Intelligence Assignment 2B.

## Overview
This repository contains the source code for the Traffic-Based Route Guidance System (TBRGS). The system integrates machine learning (ML) models for predicting traffic volume in the Boroondara area with graph search algorithms to recommend optimal travel routes based on estimated travel times.
### Key Features
* **Traffic Prediction:** Uses Deep Learning models (LSTM, GRU, and [3rd Algorithm TBD]) to forecast traffic flow at SCATS intersections based on VicRoads data.
* **Travel Time Estimation:** Dynamically calculates edge costs (travel time) based on the predicted traffic flow and distance.
* **Pathfinding Integration:** Integrates search algorithms from Assignment 2A (BFS, DFS, GBFS, A*, CUS1, CUS2) alongside a top-k NetworkX implementation.
* **Interactive GUI:** A fully-featured graphical interface built with `customtkinter` featuring map visualization, panning, zooming, and results comparison.

## 🛠️ Prerequisites & Setup
To run this project, you strictly NEED **Python 3.11.15** installed on your machine, or else tensorflow dependencies might not work correctly.
**It is highly recommended to use the `uv`** virtual environment if you already have another Python version installed on your machine. `uv` will automatically download and use the correct Python version for this project without affecting your system's global installation.
### Installation
1\. **Clone the repository:**

```bash
git clone https://github.com/Steward605/COS30019-IntroductionToAI-Assignment2B.git
cd COS30019-IntroductionToAI-Assignment2B
```

2\. **Create a Virtual Environment using uv (Highly Recommended):**

**Install `uv` first** if you haven't already via `pip install uv`. Then, run the following to create an isolated environment specifically using **Python 3.11.15**:
```bash
uv venv --python 3.11.15
```

3\. **Activate the Virtual Environment:**

On Windows:
```bash
.venv\Scripts\activate
```

4\. **Install Dependencies:**

Install the required data processing, machine learning, and visualization dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage Guide
To ensure the system works correctly, the pipeline must be executed in the following order:
1\. **Data Processing**

First, process the raw traffic data to prepare it for the machine learning models.
```bash
python data-process.py
```
***What this does:** This script flattens the sequence, chronologically splits the data into training/testing sets, applies a scaler, and generates the sliding window 3D tensors required by TensorFlow/Keras.*

2\. **Model Training**

Next, train the machine learning algorithms on the processed data. You only need to run this once (or whenever you adjust the model architecture).
```bash
python train-lstm.py
python train-gru.py
```
***What this does:** These scripts train the LSTM and GRU models. Running them will automatically create a models/ directory for the .keras files and a logs/ directory to safely store the evaluation plots and training logs without cluttering the root folder.*

3\. **Route & Graph Preparation**

Run the graph preparation script to construct the road network.
```bash
python prepare_scats_graph.py
```

4\. **Launch the Route Guidance System**

Finally, launch the interactive GUI to input origins, destinations, and visualize the optimal routes on the map.
```bash
python gui.py
```