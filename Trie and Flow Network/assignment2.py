"""
Author: Jessica Angeline Tjandra - 32976712 
FIT2004 S2 2023 Assignment 2
                            Trie and Flow Network
"""

from math import ceil, inf
### DO NOT CHANGE THIS FUNCTION
def load_dictionary(filename):
    infile = open(filename)
    word, frequency = "", 0
    aList = []
    for line in infile:
        line.strip()
        if line[0:4] == "word":
            line = line.replace("word: ","")
            line = line.strip()
            word = line            
        elif line[0:4] == "freq":
            line = line.replace("frequency: ","")
            frequency = int(line)
        elif line[0:4] == "defi":
            index = len(aList)
            line = line.replace("definition: ","")
            definition = line.replace("\n","")
            aList.append([word,definition,frequency])

    return aList

class Node:
    """ Node class representing each node in the trie """
    def __init__(self, size=27) -> None: 
        """
        Function description: initialise the node object
        Approach description (if main function): 
        
        :Input:
        size: integer represent the size of the array of the node's links of indices 
                by default it has 1-26 (26 letters of alphabet) and 0 for terminal $
        
        :Output, return or postcondition: Node object is initialised
        :Time complexity: O(1) + O(size=27), where size is the input size
        :Aux space complexity: O(size=27), where size is the input size
        """
        # data [word, definition, frequency]
        self.word = ""
        self.definition = ""
        
        # terminal $ at index 0
        self.links = [None] * size # 26 letters + 1 for $

        self.name = ""    #char  
        self.highest_priority_id = inf
        self.highest_priority_frequency = 0
        self.num_matches = 0      


class Trie:
    """ Trie class representing the trie data structure"""
    def __init__(self, Dictionary):
        """ create a Trie based on the Dictionary which is a list
        of lists produced by load_dictionary function as described above.
        refer to the lecture notes on how to create a trie: https://youtu.be/1qFlSsDKEYI

        Function description: initialise the Trie object
        Approach description (if main function): break down each item in the dictionary into word, definition and frequency
                                                and insert them into the trie
        :Input:
        Dictionary: list of lists produced by load_dictionary function as described above

        :Output, return or postcondition: Trie object is initialised
        :Time complexity: O(T)
        :Aux space complexity: O(T)
        where T is the total number of characters in Dictionary.txt.
        """
        self.root = Node()

        for word, definition, frequency in Dictionary:
            self.insert(word, definition, frequency)


    def insert(self, word, definition, frequency):
        """
        Function description: initialise the Trie object
        Approach description (if main function): insert word, its definition, and its frequency into the trie
        :Input:
        word: string representing the word
        definition: string representing the definition of the word
        frequency: integer representing the frequency of the word

        :Output, return or postcondition: word, definition and frequency are inserted into the trie
        :Time complexity: O(N)
        :Aux space complexity: O(N)
        N is the total number of characters in the word with the highest frequency and its definition.        
        """
        # begin from root
        current = self.root

        #go thru key 1 by 1
        for char in word:
            #get index
            index = ord(char) - ord('a') + 1

            #if link doesnt exist, create it
            if current.links[index] == None:
                current.links[index] = Node()
                current.links[index].name = char

            if frequency > current.highest_priority_frequency:
                current.highest_priority_id = index
                current.highest_priority_frequency = frequency

            elif frequency == current.highest_priority_frequency:
                if index < current.highest_priority_id:
                    current.highest_priority_id = index
                    current.highest_priority_frequency = frequency

            
            #move to next node
            current.num_matches += 1
            current = current.links[index]
           

        # last letter
        if frequency > current.highest_priority_frequency:
            current.highest_priority_id = 0
            current.highest_priority_frequency = frequency
        current.num_matches += 1

        # mark end of word
        terminal = Node()
        terminal.word = word
        terminal.definition = definition
        terminal.name = "$"

        index = 0
        if current.links[index] is None:
            current.links[index] = terminal


    def prefix_search(self, prefix):
        # ToDo: this function must take a prefix as input and return a list
        """"
        [word, definition, num_matches] containing three elements where
        the first element "word" is the prefix matched word with the highest
        frequency, "definition" is its definition from the dictionary and
        num_matches is the number of words in the dictionary that have
        the input prefix as their prefix 
        
        
        Function description: search for the prefix in the trie
        Approach description (if main function): go thru the prefix and check if the prefix exists in the trie
        :Input:
        prefix: string representing the prefix
        
        :Output, return or postcondition: a list containing the word, definition and num_matches
        :Time complexity: O(M + N)
        :Aux space complexity: O(1) 
         where M is the length of the prefix entered by the user and N is the total number of 
         characters in the word with the highest frequency and its definition.
        """

        current = self.root #Aux space complexity: O(1)
        for char in prefix:
            index = ord(char) - ord('a') + 1
            if current.links[index] is None:
                return [None, None, 0]
            
            current = current.links[index]

        return self.get_priority_child(current, current.highest_priority_frequency, prefix)


    def get_priority_child(self, node, highest_frequency, prefix):
        """
        Function description: get the highest priority child of the input node
        Approach description (if main function): go through the node's children and get the highest priority child

        :Input:
        node: Node object representing the node
        highest_frequency: integer representing the highest frequency of the node
        prefix: string representing the prefix

        :Output, return or postcondition: a list containing the word, definition and num_matches
        :Time complexity: O(N)
        :Aux space complexity: O(1)
        N is the total number of characters in the word with the highest frequency and its definition.
        """
        last_node = node

        if prefix == "":
            max_id = self.root.highest_priority_id
            last_node = self.root.links[max_id]

        while node.highest_priority_frequency == highest_frequency:
            node = node.links[node.highest_priority_id]
        if prefix == "":
            return [node.word, node.definition, self.root.num_matches]
        if node.name == "$":
            return [node.word, node.definition, last_node.num_matches]
           
                
    def _str_helper(self, node, prefix):
        """
        Function description: helper function for __str__ function
        Approach description (if main function): recursively go through the trie and print the words in the trie

        :Input:
        node: Node object representing the node
        prefix: string representing the prefix

        :Output, return or postcondition: a string representing the words in the trie
        :Time complexity: O(N)
        :Aux space complexity: O(N)
        N is the total number of characters in the word with the highest frequency and its definition.
        """
        if node.data is not None:
            prefix += f' -> {node.name}'
        for link in node.links:
            if link is not None:
                prefix = self._str_helper(link, prefix)
        return prefix

    def __str__(self):
        return self._str_helper(self.root, '')

# ------------------------------------------------
#                       q2
# ------------------------------------------------

'''adjacency list'''
class ResidualGraph:
    def __init__(self, preferences, licences) -> None:
        '''
        This function initialises the graph object.

        This is referenced from the lecture notes in FIT2004
        written by Jessica Angeline Tjandra

        Precondition: paths is a valid list of paths
        Postcondition: graph is initialised

        Input:
            preferences: list of lists representing the preferences where each sublist represents a person's preferences
            licences: list of integers representing the people with car licenses
        Return:
            None

        Time complexity:
            Best == Worst: O(|V|+ |E|), where E is the number of edges and |V| is the number of vertices.
        Space complexity:
            Aux: O(|V|+ |E|), where |V| is the number of vertices and where |E| is the number of edges.
        '''
        self.preferences = preferences
        self.licences = licences
        self.no_cars = ceil(len(preferences)/5) # 2 --> 0, 1


        # find the number of vertices needed for the outer array
        self.max_people_vertex_id = len(preferences) - 1
        
        #total vertices_id with cars license and without
        self.max_vertex_id = self.max_people_vertex_id + (self.no_cars * 2)
        
        # combine cars with license and without
        self.max_vertex_id += self.no_cars
 
        # make source  & sink
        self.max_vertex_id += 3
        self.source = self.max_vertex_id - 2
        self.sink = self.max_vertex_id - 1

        self.super_source = self.max_vertex_id

        self.cars_starting_id = self.max_people_vertex_id + 1
        self.combined_cars_starting_id = self.cars_starting_id + (self.no_cars * 2)


        self.paths = self.make_cars_ppl_paths(preferences, licences)
       
       # create the outer array
        self.vertices = [None] * (self.max_vertex_id + 1)  #+1 ensures the last vertex is included
        for vertex_id in range(self.max_vertex_id + 1):
            self.vertices[vertex_id] = Vertex(vertex_id)
        
        # add the edges
        self.add_edges(self.paths)
        self.make_cars_result_paths()

        self.make_src_sink_edges()


    def reset(self):
        """
        This function resets the vertices' attributes.
        it is used to reset once dijikstra is performed so that we can perform second time.
        written by Jessica Angeline Tjandra

        Precondition: None
        Postcondition: vertices' attributes are reset

        Input:
            None
        Return:
            None

        Time complexity:
            Best == Worst: O(|V|), where |V| is the number of vertices.
        Space complexity:
            Input: O(1)
            Aux: O(1)
        """
        for i in range(len(self.vertices)):
            self.vertices[i].discovered = False
            self.vertices[i].visited = False
            # self.vertices[i].previous = None
            self.vertices[i].min_flow = inf
    
    def add_edges(self, paths):
        """
        This function adds the edges to the vertices' edge list.
        referenced from the lecture notes in FIT2004
        written by Jessica Angeline Tjandra

        Precondition: paths is a valid list of paths
        Postcondition: edges are added to the vertices' edge list

        Input:
            None
        Return:
            None

        Time complexity:
            Best == Worst: O(|E|), where E is the number of edges.
        Space complexity:
            Input: O(|E|), where E is the number of edges.
            Aux: O(|E|), where E is the number of edges.
        """

        for u, v, r in paths:
            edge = Edge(u, v, r)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(edge)

            # add the reverse edge
            reversed_edge = Edge(v, u, 0) 
            current_reversed_vertex = self.vertices[v]
            current_reversed_vertex.add_edge(reversed_edge)

            # stores the reverse edge in the edge object
            edge.reverse_edge = reversed_edge
            reversed_edge.reverse_edge = edge

    def make_cars_ppl_paths(self, preferences, licenses):
        """
        Function description: make the paths for the cars and people
        Approach description (if main function): loop through the preferences and licenses and make the paths
                                                according to the each person's preferences and licenses
        :Input:
        preferences: list of lists representing the preferences where each sublist represents a person's preferences
        licences: list of integers representing the people with car licenses

        :Output, return or postcondition: a list of tuples representing (u, v, c) where u is the start vertex, v is the end vertex, and c is the capacity
        :Time complexity: O(N^2)
        :Aux space complexity: O(N^2)
        where N is the number of people.
        """
        paths = []

        #connect the people to the normal preferences
        for i in range(len(preferences)):
            if len(preferences[i]) == 1:
                paths.append((preferences[i][0] + self.cars_starting_id, i, 1)) #make a tuple of paths (pref, ppl, capacity)
            else:
                for pref in preferences[i]:
                    paths.append((pref + self.cars_starting_id, i, 1))
        
        #connect the people to the cars with licenses
        for person in licenses:
            if len(preferences[person]) == 1:
                paths.append((preferences[person][0] + self.no_cars  +  self.max_people_vertex_id + 1, person, 1)) #make a tuple of paths (pref, ppl, capacity)
            else:
                for pref in preferences[person]: #copy of the cars with license demands
                    paths.append((pref + self.no_cars +  self.max_people_vertex_id + 1, person, 1))

    
        return paths
    
    def make_cars_result_paths(self):
        """
        Function description: make the paths for the result car to the dummy cars of non-license cars and license cars

        Approach description (if main function): loop through the number of cars needed and make the paths

        :Input:
        None

        :Output, return or postcondition: a list of tuples representing (u, v, c) where u is the start vertex, v is the end vertex, and c is the capacity

        :Time complexity: O(ceil(N/5)) + O(self.add_edges)
        :Aux space complexity: O(ceil(N/5)) + O(self.add_edges)
        where N is the number of people.
        """


        xy_outgoing = []

        for i in range(self.no_cars):
            xy_outgoing.append((self.combined_cars_starting_id + i, self.cars_starting_id + i, 3))
            xy_outgoing.append((self.combined_cars_starting_id + i, self.cars_starting_id + i + self.no_cars, 3))
        
         
        for each_combined_car in range(self.combined_cars_starting_id, self.source):
            xy_outgoing.append((each_combined_car, self.sink, 2))

        self.add_edges(xy_outgoing)


    def make_src_sink_edges(self):
        """
        Function description: make the paths for the source and sink

        Approach description (if main function): loop through the number of cars needed and make the paths from demand vertex to cars
                                                loop through the number of people and make the paths to sink
                                                loop through the number of cars needed and make the paths from source to cars and vertex with negative demands

        :Input: 
        None

        :Output, return or postcondition: a list of tuples representing (u, v, c) where u is the start vertex, v is the end vertex, and c is the capacity

        :Time complexity: O(N) + O(ceil(N/5)) + O(self.add_edges)
        :Aux space complexity: O(N) + O(ceil(N/5)) + O(self.add_edges)
        where N is the number of people.
        """
        dummy_edges = []

        # Connect to people sink
        for person in range(len(self.preferences)):
            dummy_edges.append((person, self.sink, 1)) 

        y_upper = self.no_cars * 2

        # Connect source to cars
        for each_combined_car in range(self.combined_cars_starting_id, self.source):
            dummy_edges.append((self.source, each_combined_car, 5))

        # connect source to accomodate demands for licenses
        for each_car in range(self.cars_starting_id + self.no_cars, self.combined_cars_starting_id):
            dummy_edges.append((self.super_source, each_car, 2))
        dummy_edges.append((self.super_source, self.source, len(self.preferences)))

        # sink to source
        dummy_edges.append((self.sink, self.super_source, len(self.preferences) + y_upper))
      
        self.add_edges(dummy_edges)

    def bfs(self):
        """
        This function performs breadth first search on the graph.
        written by Jessica Angeline Tjandra
        referenced from the lecture notes in FIT2004: https://youtu.be/_EVgZwKLfZg

        Precondition: None
        Postcondition: vertices are marked as discovered

        Input:
            None
        Return:
            None

        Time complexity:
            Best == Worst: O(|V|+ |E|), where E is the number of edges and |V| is the number of vertices.
        Space complexity:   
            Aux: O(|V|), where |V| is the number of vertices.
                    """
        # initialise queue
        self.reset()
        queue = []
        source = self.vertices[self.super_source]
        queue.append(source)
        source.added_to_queue()

        # while queue is not empty
        while len(queue) > 0:
            # dequeue
            current_vertex = queue.pop(0)
            current_vertex.visit_node()

            # when it reaches sink --> return true, end BFS early
            if current_vertex.id == self.sink:
                return True 

            # for each edge
            for edge in current_vertex.edges:
                
                # if edge is not full and the vertex is not discovered
                if edge.residual > 0 and self.vertices[edge.v].discovered == False and self.vertices[edge.v].visited == False:
                    # add to queue
                    queue.append(self.vertices[edge.v])
                    self.vertices[edge.v].added_to_queue()
                    self.vertices[edge.v].previous = edge

        return False


    def ford_fulkerson(self):
        """
        This function performs the Ford Fulkerson algorithm on the graph.
        written by Jessica Angeline Tjandra

        Precondition: None
        Postcondition: max flow is returned

        Input:
            None
        Return:
            max_flow: integer representing the max flow
            all_paths: list of lists representing the paths

        Time complexity:  O(F E) ~ O(V E^2)
        where F is the max flow
                - because you need to run it F times
                - flow increment each time
                    - minimum increment is 1
            - E is the complexity of running BFS or DFS
                - because you need to run it E times, each time is O(V+E)
                - each time you need to find a path
                    - path length is at least 1
        Space complexity:
            Input: O(1)
            Aux: O(|V|) from self.bfs and + O(N) from self.get_path
        
        where N is the number of people and |V| is the number of vertices.
        """

        max_flow = 0    

        while self.bfs():
            v = self.sink

            bottleneck = inf

            while v != self.super_source:
                u_id = self.vertices[v].previous.u
                
                bottleneck = min(bottleneck, self.vertices[v].previous.residual)
                v = u_id

            max_flow += bottleneck

            # update residual capacities of the edges and reverse edges along the path
            v = self.sink
            
            while v != self.super_source:
                u_prev_edge = self.vertices[v].previous
                reverse_edge = u_prev_edge.reverse_edge

                # regular graph's capacity decreases by the path flow
                u_prev_edge.residual -= bottleneck
                # reverse graph's capacity increases by the path flow
                reverse_edge.residual += bottleneck

                v = self.vertices[v].previous.u


        path = self.get_path()
        return max_flow, path
    
    def get_path(self):
        """
        This function gets the path from the graph.
        written by Jessica Angeline Tjandra

        Precondition: None
        Postcondition: path is returned

        Input:
            None
        Return:
            path: list of lists representing the paths

        Time complexity:
            Best == Worst: O(N)
        Space complexity:
            Aux: O(N)
        where N is the number of people.
        """
        path = [[] for car in range(self.no_cars)]

        for car in range(self.cars_starting_id, self.combined_cars_starting_id):
            for edge in self.vertices[car].edges:
                if edge.residual == 0 and edge.v <= self.max_people_vertex_id:
                    if car - len(self.preferences) < self.no_cars:
                        path[car - self.cars_starting_id].append(edge.v)
                    else:
                        path[(car - self.cars_starting_id) - self.no_cars].append(edge.v)


        return path
        
class Vertex:
    """Vertex class representing each vertex in the graph"""
    def __init__(self, id) -> None:
        '''
        This function initialises the vertex object.
        
        Precondition: id is a valid id (0 <= id < len(self.vertices))
        Postcondition: vertex is initialised
        
        Input:
            id: integer representing the vertex id
            
        Return:
            None
            
        Time complexity:
            Best == Worst: O(1)
            
        Space complexity:
            Input: O(1)
            Aux: O(1)
            '''
        #linked list of edges
        self.id = id
        self.edges = []

        # for traversal
        self.discovered = False
        self.visited = False
        self.min_flow = inf

        # storing the edge (could be reverse)
        self.previous = None

        # chosen car
        self.chosen_car_id = 0


    def add_edge(self, edge):
        '''
        This function adds an edge to the vertex's edge list.
        
        Precondition: edge is a valid edge
        Postcondition: edge is added to the vertex's edge list
        
        Time complexity:
            Best == Worst: O(1)
            
        Space complexity:
            Input: O(1)
            Aux: O(1)
        '''
        self.edges.append(edge)


    def added_to_queue(self):
        '''
        This function marks the vertex as discovered.

        Precondition: None
        Postcondition: vertex.discovered = True
        
        Time complexity:
            Best == Worst: O(1)
        Space complexity:
            Input: O(1)
            Aux: O(1)
        
        '''        
        self.discovered = True

    def visit_node(self):
        '''
        This function marks the vertex as visited.

        Precondition: None
        Postcondition: vertex.visited = True
        
        Time complexity:
            Best == Worst: O(1)
        Space complexity:
            Input: O(1)
            Aux: O(1)
        
        '''
        self.visited = True

class Edge:
    """Edge class representing each edge in the graph"""
    def __init__(self, u, v, r) -> None: 
        '''
        This function initialises the edge object.

        Precondition: u, v, r are valid values
        Postcondition: edge is initialised

        Input:
            u: integer representing the start vertex id
            v: integer representing the end vertex id
            r: integer representing the residual value of this edge

        Return:
            None
        
        Time complexity:
            Best == Worst: O(1)
        Space complexity:
            Input: O(1)
            Aux: O(1)
        '''

        self.u = u
        self.v = v
        self.residual = r

        self.reverse_edge = None


def allocate(preferences, licences):
    """
    Function description: allocate the cars to the people based on their preferences and licenses 
                            on a graph with resolved lower bound and demands using ford fulkerson with bfs
    Approach description (if main function): use the ford fulkerson algorithm to allocate the cars to the people

    :Input:
    preferences: list of lists representing the preferences where each sublist represents a person's preferences
    licences: list of integers representing the people with car licenses

    :Output, return or postcondition: a list of lists representing the paths

    :Time complexity Worst: O(N^3)
    :Aux space complexity Worst: O(N^3)
    where N is the number of people.
    """
    sum_cars = ceil(len(preferences)/5)
    if ([] in preferences or 
        len(preferences) == 0 or 
        len(licences) < sum_cars*2):
        return None

    # check too many same preferences
    for car in range(sum_cars):
        count = 0
        for person in preferences:
            if car in person:
                count += 1
        if count < 2:
            return None
 

    residual_graph = ResidualGraph(preferences, licences)

    max_flow, all_paths = residual_graph.ford_fulkerson()


    for path in all_paths:
        if len(path) < 2:
            return None
    

    return all_paths

      