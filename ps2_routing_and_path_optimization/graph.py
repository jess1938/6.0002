# 6.0002 Problem Set 2 Spring 2022
# Graph Optimization
# Name:
# Collaborators:
# Time:


# This file contains a set of data structures to represent the graphs
# that you will be using for this pset.

class Node():
    """Represents a node in the graph"""

    def __init__(self, name):
        """
        Initializes an instance of Node object.

        Parameters:
            name: object
                The name of the node.
        """
        self.name = str(name)

    def get_name(self):
        """
        Returns:
            str
                The name of the node.
        """
        return self.name

    def __str__(self):
        """
        This is the function that is called when print(node) is called.

        Returns:
            str
                Humanly readable reprsentation of the node.
        """
        return self.name

    def __repr__(self):
        """
        Formal string representation of the node.

        Returns:
            str
                The name of the node.
        """
        return self.name

    def __eq__(self, other):
        """
        This is function called when you use the "==" operator on nodes.

        Parameters:
            other: Node
                Node object to compare against.

        Returns:
            bool
                True if self == other, False otherwise.
        """
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __ne__(self, other):
        """
        This is function called when you used the "!=" operator on nodes.

        Parameters:
            other: Node
                Node object to compare against.

        Returns:
            bool
                True if self != other, False otherwise.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """
        Returns:
            int
                Hash of the node. This function is necessary so that Nodes can be
                used as keys in a dictionary, Nodes are immutable.
        """
        return self.name.__hash__()


# PROBLEM 1: Implement this class based on the given docstring.
class DirectedRoad():
    """Represents a road (edge) with a travel time (weight)"""

    def __init__(self, src_node, dest_node, travel_time, road_type, traffic_multiplier):
        """
        Initialize src_node, dest_node, travel_time, road_type, traffic_multiplier for the DirectedRoad class

        Parameters:
            src_node: Node
                The source node.
            dest_node: Node
                The destination node.
            travel_time: float
                The time travelled between the src and dest.
            road_type: str
                The type of road of the edge.
            traffic_multiplier: float
                How much to multiply travel_time in the case of traffic.
                The traffic multiplier will always be at least 1.
        """
        self.src_node = src_node 
        self.dest_node = dest_node
        self.travel_time = travel_time
        self.road_type = road_type
        self.traffic_multiplier = traffic_multiplier


    def get_source_node(self):
        """
        Getter method for DirectedRoad.

        Returns:
            Node
                The source node.
        """
        return self.src_node 


    def get_destination_node(self):
        """
        Getter method for DirectedRoad.

        Returns:
            Node
                The destination node.
        """

        return self.dest_node 

    def get_road_type(self):
        """
        Getter method for DirectedRoad.

        Returns:
            str
                The road type of the road.
        """
        return self.road_type

    def get_travel_time(self, has_traffic=False):
        """
        Gets the travel_time for this road. If there is traffic,
        - multiply the time it takes to travel on a road by its traffic multiplier.

        Parameter:
            has_traffic: bool
                True if there is traffic, False otherwise.

        Returns:
            float
                The time to travel from the source node to the destination node.
        """
        if has_traffic:
            return self.travel_time*self.traffic_multiplier 
        else:
            return self.travel_time
        

    def get_traffic_multiplier(self):
        """
        Getter method for DirectedRoad.

        Returns:
            float
                The traffic multiplier.
        """

        return self.traffic_multiplier

    def __str__(self):
        """
        Function that is called when print() is called on a DirectedRoad object.

        Returns:
            str
                With the format
                'src -> dest takes travel_time minute(s) via road_type road with traffic multiplier traffic_multiplier'

        Note: For the total time assume normal traffic conditions.
        """

        return "{} -> {} takes {} minute(s) via {} road with traffic multiplier {}".format(self.src_node.get_name(), self.dest_node.get_name(), self.travel_time, self.road_type, self.traffic_multiplier)
           

    def __hash__(self):
        """
        Returns:
            int
                Hash of the road. This function is necessary so that DirectedRoads can be
                used as keys in a dictionary, DirectedRoads are immutable.
        """
        return self.__str__().__hash__()

# PROBLEM 1: Implement methods of this class based on the given docstring.
# DO NOT CHANGE THE FUNCTIONS THAT HAVE BEEN IMPLEMENTED FOR YOU.
class RoadMap():
    """Represents a road map -> a directed graph of Node and DirectedRoad objects"""

    def __init__(self):
        """
        Initalizes a new instance of RoadMap.
        """

        self.nodes = set()
        self.nodes_to_roads = {}  # must be a dictionary of Node -> list of roads starting at that node

    def __str__(self):
        """
        Function that is called when print() is called on a RoadMap object.

        Returns:
            str
                Representation of the RoadMap.
        """

        road_strs = []
        for roads in self.nodes_to_roads.values():
            for road in roads:
                road_strs.append(str(road))
        road_strs = sorted(road_strs)  # sort alphabetically
        return '\n'.join(road_strs)  # concat road_strs with "\n"s between them

    def get_all_nodes(self):
        """
        Returns:
            set of Node
                A COPY of all nodes in the RoadMap. Does not modify self.nodes.
        """
        node_copy = set()
        for node in self.nodes: 
            node_copy.add(node)
        return node_copy 
    
    def contains_node(self, node):
        """
        Parameter:
            node: Node, node to check

        Returns:
            bool
                True, if node is in the graph; False, otherwise.
        """
        return node in self.nodes
    

    def insert_node(self, node):
        """
        Adds a Node object to the RoadMap.
        Raises a ValueError if it is already in the graph.

        Parameter:
            node: Node
                Node to add.
        """
        if not (node in self.nodes):
            self.nodes.add(node)
            self.nodes_to_roads[node] = []
        else:
            raise ValueError
            

    def insert_road(self, road):
        """
        Adds a DirectedRoad instance to the RoadMap.
        Raises a ValueError if either of the nodes associated with the road is not in the graph.

        Parameter:
            road: DirectedRoad
                Road to add.
        """
        src_n = road.get_source_node()
        dest_n = road.get_destination_node()
        
        if self.contains_node(src_n) and self.contains_node(dest_n):
            self.nodes_to_roads[src_n].append(road)
            #self.nodes_to_roads[dest_n].append(road)
            
        else:
            raise ValueError 

    def get_reachable_roads_from_node(self, node, restricted_roads):
        """
        Gets the roads out of Node node, excluding roads whose types are in restricted_roads.

        Parameters:
            node: Node
                Find reachable roads out of this node.
            restricted_roads: list of str (types of roads)
                Road types that cannot be traveled on.

        Returns:
            list of DirectedRoad
                A new list of all the roads that start at given node, whose types are not in restricted_roads.
                Empty list if the node is not in the graph.
        """
        reachable_roads = []
        if restricted_roads != None:
            if self.contains_node(node):
                #print(self.nodes_to_roads[node])
                for road in self.nodes_to_roads[node]:
                    #print(restricted_roads)
                    if road.get_road_type() not in restricted_roads:
                        reachable_roads.append(road)
        else:
            return self.nodes_to_roads[node]
        return reachable_roads

# if __name__ == '__main__':
#     # node1 = Node("node1")
#     # node2 = Node("node2")
#     # road1 = DirectedRoad(node1, node2, 3.0, 'highway', 2.0)
#     # map1 = RoadMap()
#     # map1.insert_node(node1) 
#     # map1.insert_node(node2)
#     # map1.insert_road(road1)
#     # map1.get_reachable_roads_from_node(node1, [road1])
