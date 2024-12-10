# author: Soma S Dhavala
# date: 25th December, 2019
# Ref: https://docs.google.com/document/d/1vyeHU3ahDZ-4WOnmpu6f02QTfPh3ngbq8rZIJ2o3zUI/edit?usp=sharing

import networkx as nx
import numpy as np
import time
from datetime import datetime, timedelta

# simulate a complete graph db, with 10 nodes
n = 10

# want to go from src to dst
src = 0
dst = 9


max_dist = 100
stop = False
counter = 0
missing_links = [[src,dst]]


def SimulateData(n=10):
	db = nx.complete_graph(n)
	for e in list(db.edges):
		db.edges[e]['dist'] = np.random.randint(1,5)
	return db



class Route():
	def __init__(self, src, dst, edge_length=0, edge_duration=0, edge_cost=0):
		self.src = src
		self.dst = dst
		self.edge_length = edge_length
		self.edge_duration = edge_duration
		self.edge_cost = edge_cost
		self.is_transfer = False
          
class StoppingCriteria:
    def __init__(self, max_iterations=10, time_limit_ms=5000, min_paths=5):
        self.max_iterations = max_iterations  # S1
        self.time_limit_ms = time_limit_ms   # S2
        self.min_paths = min_paths           # S3
        self.start_time = None
    
    def is_met(self, current_iteration, num_paths):
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        return (current_iteration >= self.max_iterations or
                elapsed_ms >= self.time_limit_ms or
                num_paths >= self.min_paths)

class Vertex:
    def __init__(self, id, lat, lon):
        self.id = id
        self.lat = lat
        self.lon = lon

class Origin(Vertex):
    def __init__(self, id, lat, lon, expected_departure_time, departure_buffer):
        super().__init__(id, lat, lon)
        self.expected_departure_time = expected_departure_time
        self.departure_buffer = departure_buffer

class Destination(Vertex):
    def __init__(self, id, lat, lon, expected_arrival_time, arrival_buffer):
        super().__init__(id, lat, lon)
        self.expected_arrival_time = expected_arrival_time
        self.arrival_buffer = arrival_buffer

class RuleBase:
    def __init__(self):
        self.edge_rules = {
            'E1': lambda edge: edge.edge_length <= self.thresholds['edge_length'],
            'E2': lambda edge: edge.edge_duration <= self.thresholds['edge_duration'],
            'E3': lambda edge: edge.edge_cost <= self.thresholds['edge_cost'],
            'E4': lambda edge: edge.edge_centrality >= self.thresholds['edge_centrality']
        }
        
        self.path_rules = {
            'P1': lambda path: path.total_length <= self.thresholds['path_length'],
            'P2': lambda path: path.total_duration <= self.thresholds['path_duration'],
            'P3': lambda path: path.total_cost <= self.thresholds['path_cost'],
            'P4': lambda path: path.num_transfers <= self.thresholds['path_transfers']
        }
        
        self.stopping_rules = {
            'S1': lambda iter: iter <= self.thresholds['max_iterations'],
            'S2': lambda time: time <= self.thresholds['time_limit_ms'],
            'S3': lambda paths: len(paths) >= self.thresholds['min_paths']
        }

class Journey:
    def __init__(self, path, length, duration, cost, transfers):
        self.path = path
        self.length = length
        self.duration = duration
        self.cost = cost
        self.transfers = transfers
	
# this is to be implemented by service providers
def GetRoutes(missing_links):
	routes = []
	for link in missing_links:
		src, dst = link[0], link[1]
		
		# If we have string IDs, we need to create temporary vertices with actual coordinates
		if isinstance(src, str):
			# Try to get coordinates from the original vertex if available
			src = Origin(
				id=src,
				lat=12.9716,  # Default Bangalore coordinates
				lon=77.5946,
				expected_departure_time=datetime.now(),
				departure_buffer=15
			)
		if isinstance(dst, str):
			dst = Destination(
				id=dst,
				lat=13.0827,  # Default Bangalore coordinates
				lon=77.5877,
				expected_arrival_time=datetime.now() + timedelta(hours=1),
				arrival_buffer=10
			)
			
		edge_length = calculate_heuristic_distance(src, dst)
		tmp = Route(
			src=src.id,  # Store IDs in the route
			dst=dst.id,
			edge_length=edge_length,
			edge_duration=edge_length/30,  # Assume 30 km/h
			edge_cost=edge_length*10       # Assume 10 INR per km
		)
		routes.append(tmp)
	return routes


def UpdateGraph(G, routes):
	for route in routes:
		src = route.src
		dst = route.dst
		# Add nodes if they don't exist
		if src not in G:
			G.add_node(src)
		if dst not in G:
			G.add_node(dst)
		# Add or update edge
		G.add_edge(src, dst)
		G[src][dst].update({
			'edge_length': route.edge_length,
			'edge_duration': route.edge_duration,
			'edge_cost': route.edge_cost,
			'is_transfer': route.is_transfer
		})
	return G

def FilterRankSelectPaths(paths, G, rules=None):
    if rules is None:
        rules = {
            'max_length': 100,
            'max_duration': 24,
            'max_cost': 1000,
            'max_transfers': 3
        }
    
    valid_paths = []
    for path in paths:
        # Check if path meets all rules
        if (path['length'] <= rules['max_length'] and
            path['duration'] <= rules['max_duration'] and
            path['cost'] <= rules['max_cost'] and
            path['transfers'] <= rules['max_transfers']):
            # Convert dictionary to Journey object
            journey = Journey(
                path=path['path'],
                length=path['length'],
                duration=path['duration'],
                cost=path['cost'],
                transfers=path['transfers']
            )
            valid_paths.append(journey)
    
    # Rank paths by multiple criteria
    ranked_paths = sorted(valid_paths, 
                         key=lambda x: (x.transfers, x.duration, x.cost))
    
    # Select top K paths
    K = 5
    return ranked_paths[:K]

def FilterRankSelectMissingLinks(G, rules=None):
    """
    Implementation following README.md FilterRankSelectMissingLink function
    """
    if rules is None:
        rules = {
            'max_length': 50,      # E1
            'max_duration': 12,     # E2
            'max_cost': 500,       # E3
            'min_centrality': 0.1   # E4
        }
    
    missing_links = []
    nodes = list(G.nodes())
    
    # Create a mapping of node IDs to their coordinates
    node_coords = {}
    for node in nodes:
        # If node is a string ID, create a temporary vertex with default coordinates
        if isinstance(node, str):
            node_coords[node] = Vertex(id=node, lat=0, lon=0)
        else:
            node_coords[node] = node
    
    # Find all potential missing links
    for i in nodes:
        for j in nodes:
            if i != j and not G.has_edge(i, j):
                # Get vertex objects for distance calculation
                src_vertex = node_coords[i]
                dst_vertex = node_coords[j]
                
                # Calculate heuristic distance and estimates
                est_length = calculate_heuristic_distance(src_vertex, dst_vertex)
                est_duration = est_length / 30  # Assume 30 km/h
                est_cost = est_length * 10      # Assume 10 INR per km
                
                # Filter based on rules E*
                if (est_length <= rules['max_length'] and
                    est_duration <= rules['max_duration'] and
                    est_cost <= rules['max_cost']):
                    missing_links.append({
                        'src': i,
                        'dst': j,
                        'est_length': est_length,
                        'est_duration': est_duration,
                        'est_cost': est_cost,
                        'impact_score': calculate_impact_score(G, i, j)
                    })
    
    # Rank missing links by impact score
    ranked_links = sorted(missing_links, key=lambda x: x['impact_score'], reverse=True)
    
    # Select top K missing links
    K = 5
    return [(link['src'], link['dst']) for link in ranked_links[:K]]






def process_mobility_catalogs(catalogs, G):
    """Process mobility catalogs and update graph"""
    for catalog in catalogs:
        provider_type = catalog['type']
        for service in catalog['services']:
            src = service['origin']
            dst = service['destination']
            
            # Add nodes if they don't exist
            if src not in G:
                G.add_node(src)
            if dst not in G:
                G.add_node(dst)
                
            # Add edge with service details
            G.add_edge(src, dst, 
                edge_length=service['distance'],
                edge_duration=service['duration'],
                edge_cost=service['cost'],
                provider_type=provider_type,
                provider_id=catalog['provider_id']
            )
    return G


def calculate_heuristic_distance(src_vertex, dst_vertex):
    """Calculate distance between two vertices using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2

    # Get coordinates
    lat1 = float(src_vertex.lat)
    lon1 = float(src_vertex.lon)
    lat2 = float(dst_vertex.lat)
    lon2 = float(dst_vertex.lon)

    R = 6371  # Earth's radius in kilometers

    # Convert coordinates to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def calculate_impact_score(G, src, dst):
    """Calculate impact score for a missing link based on:
    1. Number of potential paths it could complete
    2. Potential reduction in journey times
    3. Centrality of vertices
    """
    impact_score = 0
    
    # Get vertex IDs if they are Vertex objects
    src_id = src.id if hasattr(src, 'id') else src
    dst_id = dst.id if hasattr(dst, 'id') else dst
    
    try:
        # Get all possible paths that could use this link
        all_paths = list(nx.all_simple_paths(G, src_id, dst_id))
        impact_score += len(all_paths)
        
        # Calculate potential time savings
        current_shortest = nx.shortest_path_length(G, src_id, dst_id, weight='duration')
        estimated_duration = calculate_heuristic_distance(src, dst) / 30  # assume 30km/h
        if estimated_duration < current_shortest:
            impact_score += (current_shortest - estimated_duration)
        
        # Add centrality factor
        centrality = nx.betweenness_centrality(G)
        impact_score += (centrality.get(src_id, 0) + centrality.get(dst_id, 0)) / 2
    except (nx.NetworkXNoPath, nx.NetworkXError):
        # Handle case when no path exists
        impact_score = 0
    
    return impact_score

def is_self_commutable(src_vertex, dst_vertex):
    """Determine if a missing link can be self-commuted (walking, cycling, etc.)"""
    MAX_WALKING_DISTANCE = 2  # km
    distance = calculate_heuristic_distance(src_vertex, dst_vertex)
    return distance <= MAX_WALKING_DISTANCE

db = SimulateData(n)
# current state of the graph
G = db.to_directed()

if __name__ == '__main__':
    stop = False
    counter = 0
    missing_links = [[src,dst]]
    
    while not stop:
        valid_routes = []
        routes = GetRoutes(missing_links)
        UpdateGraph(routes)
        paths = list(nx.all_simple_paths(G, src, dst, cutoff=max_dist))
        journeys = FilterRankSelectPaths(paths, G)
        missing_links = FilterRankSelectMissingLinks(G)
        print('missing links',missing_links[0])
        print('journeys',journeys[0])
        counter += 1
        if counter > 10:
            stop = True

            

def main(S, D, initial_graph=None, stopping_criteria=None):
    # Initialize as per README
    time_step = 0
    V = [S.id, D.id]  # Use vertex IDs instead of objects
    E = []
    journeys = []
    
    # Create initial graph
    if initial_graph is None:
        G = nx.DiGraph()
        G.add_nodes_from(V)
    else:
        G = initial_graph.copy()
    
    # Initial missing links
    missing_links = [(S.id, D.id)]
    
    while not stopping_criteria.is_met(time_step, len(journeys)):
        routes = GetRoutes(missing_links)
        G = UpdateGraph(G, routes)
        paths = list(nx.all_simple_paths(G, S.id, D.id))
        
        path_objects = [
            {
                'path': path,
                'length': sum(G[path[i]][path[i+1]].get('edge_length', 0) for i in range(len(path)-1)),
                'duration': sum(G[path[i]][path[i+1]].get('edge_duration', 0) for i in range(len(path)-1)),
                'cost': sum(G[path[i]][path[i+1]].get('edge_cost', 0) for i in range(len(path)-1)),
                'transfers': sum(1 for i in range(len(path)-1) if G[path[i]][path[i+1]].get('is_transfer', False))
            }
            for path in paths
        ]
        
        journeys = FilterRankSelectPaths(path_objects, G)
        missing_links = FilterRankSelectMissingLinks(G)
        time_step += 1
    
    return journeys
