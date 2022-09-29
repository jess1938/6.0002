# 6.0002 Problem Set 2 Spring 2022
# Graph Optimization
# Name: Jessica Jimenez
# Collaborators: None
# Time: 8 hours 


# Finding shortest paths to drive from home to work on a road network


from graph import DirectedRoad, Node, RoadMap


# PROBLEM 2: Building the Road Network
#
# PROBLEM 2.1: Designing your Graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the times
# represented?
#
# Graph's nodes represents the destinations
# Graph's edges represents whether a driving route exists between the 2 destinations
# Times given is the time it takes to drive from one destination to the next 
#

# PROBLEM 2.2: Implementing create_graph
def create_graph(map_filename):
    """
    Parses the map file and constructs a road map (graph).

    Travel time and traffic multiplier should be each cast to a float.

    Parameters:
        map_filename : str
            Name of the map file.

    Assumes:
        Each entry in the map file consists of the following format, separated by spaces:
            source_node destination_node travel_time road_type traffic_multiplier

        Note: hill road types always are uphill in the source to destination direction and
              downhill in the destination to the source direction. Downhill travel takes
              half as long as uphill travel. The travel_time represents the time to travel
              from source to destination (uphill).

        e.g.
            N0 N1 10 highway 1
        This entry would become two directed roads; one from 'N0' to 'N1' on a highway with
        a weight of 10.0, and another road from 'N1' to 'N0' on a highway using the same weight.

        e.g.
            N2 N3 7 uphill 2
        This entry would become two directed roads; one from 'N2' to 'N3' on a hill road with
        a weight of 7.0, and another road from 'N3' to 'N2' on a hill road with a weight of 3.5.
        Note that the directed roads created should have both type 'hill', not 'uphill'!

    Returns:
        RoadMap
            A directed road map representing the given map.
    """
    final_map = RoadMap()
    f = open(map_filename)

    for line in f: 
        #split each line by a space 
        row = line.split(" ")
        src_node = Node(row[0])
        dest_node = Node(row[1])
        
        #add nodes that have not appeared on to map
        if src_node not in final_map.get_all_nodes():
            final_map.insert_node(src_node)
        if dest_node not in final_map.get_all_nodes():
            final_map.insert_node(dest_node)
        
        #make roads for 
        if row[3] != 'uphill':
            for_road = DirectedRoad(src_node, dest_node, float(row[2]), row[3], float(row[4]))
            back_road = DirectedRoad(dest_node, src_node, float(row[2]), row[3], float(row[4]))
        else:
            #if uphill decrease travel time for the way down 
            for_road = DirectedRoad(src_node, dest_node, float(row[2]), 'hill', float(row[4]))
            back_road = DirectedRoad(dest_node, src_node, float(row[2])/2, 'hill', float(row[4]))

        final_map.insert_road(for_road)
        final_map.insert_road(back_road)

    return final_map

#!! CHECK OFF: EXPLAIN PRINTOUT OF DIGAPH !!



# PROBLEM 2.3: Testing create_graph
# Go to the bottom of this file, look for the section under FOR PROBLEM 2.3,
# and follow the instructions in the handout.


# PROBLEM 3: Finding the Shortest Path using Optimized Search Method



# Problem 3.1: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer: Our objective functions is the length of time it takes to take a possible paths. 
# We want to find the shortest time to travel from one node to another. 
# Constraints are whether a path between nodes exist, if there is traffic, and the time it takes
# to travel through a path without traffic. 

# PROBLEM 3.2: Implement find_shortest_path
def find_shortest_path(roadmap, start, end, restricted_roads=None, has_traffic=False):
    """
    Finds the shortest path between start and end nodes on the road map,
    without using any restricted roads, following traffic conditions.
    If restricted_roads is None, assume there are no restricted roads.
    Use Dijkstra's algorithm.

    Parameters:
        roadmap: RoadMap
            The graph on which to carry out the search.
        start: Node
            Node at which to start.
        end: Node
            Node at which to end.
        restricted_roads: list of str or None
            Road Types not allowed on path. If None, all are roads allowed
        has_traffic: bool
            Flag to indicate whether to get shortest path during traffic or not.

    Returns:
        A two element tuple of the form (best_path, best_time).
            The first item is a list of Node, the shortest path from start to end.
            The second item is a float, the length (time traveled) of the best path.
        If there exists no path that satisfies constraints, then return None.
    """
    
    # if either start or end is not in roadmap:
    if not roadmap.contains_node(start) or not roadmap.contains_node(end):
        return None
    # if start and end are the same node:
    if start == end:
        return ([start], 0)
    
    unvisited_nodes = list(roadmap.get_all_nodes())
    visited_nodes = []
    #dictionaries with shortest time to start node 
    time_to = {node: float('inf') for node in roadmap.get_all_nodes()}
    time_to[start] = 0
    #previous node on shortest path to start node 
    predecessor = {node: None for node in roadmap.get_all_nodes()}
    
    
    while unvisited_nodes: 
        #if least travel time to an unvisited node is ​∞​, break.
        if time_to[start] == float('inf'):
            break
        #Set unvisited node with least travel time as current node.
        current = min(unvisited_nodes, key = lambda node: time_to[node])
        #If current node is end node, break.
        if current == end:
            break
        
        
        #For each neighbor of the current node:
        for road in roadmap.get_reachable_roads_from_node(current, restricted_roads):
            neighbor = road.get_destination_node()
            #if the neighbor has not been visited, visit it and update its best path and best time if neded.
            if neighbor not in visited_nodes: 
                if not has_traffic: 
                    alternative_time = time_to[current] + road.get_travel_time()
                else:
                    alternative_time = time_to[current] + road.get_travel_time()*road.get_traffic_multiplier()
                if alternative_time < time_to[neighbor]:
                    time_to[neighbor] = alternative_time 
                    predecessor[neighbor] = current 
        # Mark the current node as visited.
        unvisited_nodes.remove(current)
        

    # If there is no path between start and end:
    if predecessor[end] == None:
        return None
    
    #finding shortest path from start to end 
    path = []
    current = end 
    #loop until reach start node 
    while predecessor[current] != None:
        path.insert(0, current)
        current = predecessor[current]
     #start node has a predecessor of None! so last step is adding start node
    if path != []:
        path.insert(0, current)
    return (path, time_to[end])    # return (best path, best time) from start to end
    

# PROBLEM 4.1: Implement optimal_path_no_traffic
def find_shortest_path_no_traffic(filename, start, end):
    """
    Finds the shortest path from start to end during conditions of no traffic.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end in normal traffic.
        If there exists no path, then return None.
    """
    roadmap = create_graph(filename)
    (path, time) = find_shortest_path(roadmap, start, end, None, False)
    return path 
    

# PROBLEM 4.2: Implement optimal_path_restricted
def find_shortest_path_restricted(filename, start, end):
    """
    Finds the shortest path from start to end when local roads and hill roads cannot be used.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies constraints, then return None.
    """
    roadmap = create_graph(filename)
    (path, time) = find_shortest_path(roadmap, start, end, ['local', 'hill'], False)
    return path 


# PROBLEM 4.3: Implement optimal_path_heavy_traffic
def find_shortest_path_in_traffic_no_toll(filename, start, end):
    """
    Finds the shortest path from start to end when toll roads cannot be used and in traffic,
    i.e. when all roads' travel times are multiplied by their traffic multipliers.

    You must use find_shortest_path.

    Parameters:
        filename: str
            Name of the map file that contains the graph
        start: Node
            Node object at which to start.
        end: Node
            Node object at which to end.

    Returns:
        list of Node
            The shortest path from start to end given the aforementioned conditions.
        If there exists no path that satisfies the constraints, then return None.
    """

    roadmap = create_graph(filename)
    (path, time) = find_shortest_path(roadmap, start, end, ['toll'], True)
    return path 


if __name__ == '__main__':
    

    # UNCOMMENT THE LINES BELOW TO DEBUG OR TO EXECUTE PROBLEM 2.3
    pass

    small_map = create_graph('./maps/small_map.txt')
    #print(small_map)

    # # ------------------------------------------------------------------------
    # FOR PROBLEM 2.3
    road_map = create_graph("maps/test_create_graph.txt")
    print(road_map)
    # # ------------------------------------------------------------------------

    # start = Node('N0')
    # end = Node('N4')
    # restricted_roads = []
    # print(find_shortest_path(small_map, start, end, restricted_roads))
