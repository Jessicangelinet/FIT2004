"""
Author: Jessica Angeline Tjandra - 32976712 
FIT2004 S2 2023 Assignment 1
                            Dynamic Programming & Dijkstra's Algorithm
"""
def restaurantFinder(d,site_list):
    '''   
    This function returns a tuple (revenue, sites) where revenue is the maximum revenue that can be obtained and sites is a list of the sites that should be selected to obtain the maximum revenue.
    Approach: Dynamic Programming by storing the maximum revenue from the previous sites and the sites that should be selected to obtain the maximum revenue (memoization).
    Reference: https://www.geeksforgeeks.org/maximum-sum-such-that-no-two-elements-are-adjacent/,
    https://www.codingninjas.com/studio/library/maximum-sum-such-that-no-two-elements-are-adjacent,
    https://youtu.be/v0hT7F_n_Sw

    written by Jessica Angeline Tjandra

    Precondition: d is a valid distance parameter (d >= 0) and site_list is a valid list of size N (len(site_list) == N)
    Postcondition: maximum revenue and the sites that should be selected to obtain the maximum revenue are found

    Input:
        d: integer representing the distance parameter
        site_list: list of integers representing the annual revenue, it has size N containing annual revenue (in million dollars)

    Return:
        (revenue, sites): tuple of the maximum revenue and the sites that should be selected to obtain the maximum revenue

    Time complexity:
        Best == Worst: O(N), where N is the number of sites.
    Space complexity:
        Input & Aux: O(N), where N is the number of sites.
    '''
    # base cases
    # if there is only 1 site, then the revenue is the site's revenue
    if len(site_list) == 1:
        return (site_list[0], [1]) #selected sites starts from 1
    # if the revenue is negative or the site_list is empty, then the revenue is 0
    if max(site_list) < 0 or len(site_list) == 0:
            return (0, [])
    # if d >= len(site_list), then the revenue is the maximum revenue from 1 of the site_list
    if d >= len(site_list):
        return (max(site_list), [site_list.index(max(site_list)) + 1])
    
    # memo DP
    memo = [0] * (len(site_list) + 1) #memo[i] is the max revenue for site i

    # base cases
    memo[1] = site_list[0]
    if site_list[0] < 0:
        memo[1] = 0

    # fill in the memo's value according to the formula: memo[i] = max(memo[i-1], memo[i-d-1] + site_list[i-1])
    for site_id in range(2, len(site_list)+1):
        # if the site's revenue is negative, then it is not selected
        if site_list[site_id-1] < 0:
            memo[site_id] = 0

        exclude = memo[site_id-1] #exclude current site as it is within d distance range
        include = memo[site_id - d - 1] + site_list[site_id-1] # include current site's rvenue and exclude the sites within d distance range

        if include > exclude:
            memo[site_id] = include

        else:
            memo[site_id] = exclude

    # backtrack to get the sites that should be selected to obtain the maximum revenue
    site_index = len(site_list)
    result = []
    while site_index >= 0:
        if memo[site_index] > memo[site_index-1]:
            result.append(site_index)
            site_index -= (d+1)
        else:
            site_index -= 1

    result = result[::-1]
            
    # the max revenue is the last element in the memo
    revenue = memo[-1]
    if revenue == 0:
        result = []

    return(revenue, result)

'''adjacency list'''
class FloorGraph:
    """FloorGraph class representing the graph of each floor"""
    def __init__(self, paths, keys, invert = False) -> None:
        '''
        This function initialises the graph object.

        This is referenced from the lecture notes in FIT2004
        written by Jessica Angeline Tjandra

        Precondition: paths is a valid list of paths
        Postcondition: graph is initialised

        Input:
            paths: list of tuples representing the paths in this current floor
            keys: list of tuples representing the keys
            invert: boolean representing whether the graph is inverted or not
        Return:
            None

        Time complexity:
            Best == Worst: O(|V|+ |E|), where E is the number of edges and |V| is the number of vertices.
        Space complexity:
            Input: - keys: O(|V|), where |V| is the number of vertices.
                   - paths: O(|E|), where |E| is the number of edges.
                   - invert: O(1)
            Aux: O(|V|+ |E|), where |V| is the number of vertices and where |E| is the number of edges.
        '''
        self.invert = invert

        self.max_vertex_id = 0

        # find the number of vertices needed for the outer array
        # time complexity: O(E)
        for u, v, x in paths:
            if u > self.max_vertex_id:
                self.max_vertex_id = u
            if v > self.max_vertex_id:
                self.max_vertex_id = v

        # inverted graph has extra 1 for dummy node that connects to all exits,
        # hence, we need to add 1 to the max_vertex_id
        if invert:
            self.max_vertex_id = self.max_vertex_id + 1 

        # create the outer array
        self.vertices = [None] * (self.max_vertex_id + 1)  #+1 ensures the last vertex is included
        for vertex_id in range(self.max_vertex_id + 1):
            self.vertices[vertex_id] = Vertex(vertex_id)
        
        # add the edges
        self.add_edges(paths)

        #for inverting the graph
        self.paths = paths
        self.keys = keys
    
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
            self.vertices[i].previous = None
            self.vertices[i].time = 0
    
    def add_edges(self, paths):
        """
        This function adds the edges to the vertices' edge list.
        written by Jessica Angeline Tjandra

        Precondition: paths is a valid list of paths
        Postcondition: edges are added to the vertices' edge list

        Input:
            paths: list of tuples representing the paths
        Return:
            None

        Time complexity:
            Best == Worst: O(|E|), where E is the number of edges.
        Space complexity:
            Input: O(|E|), where E is the number of edges.
            Aux: O(|E|), where E is the number of edges.
        """

        for u, v, x in paths:
            if self.invert:
                edge = Edge(v, u, x)
                current_vertex = self.vertices[v]
                current_vertex.add_edge(edge)
            else:
                edge = Edge(u, v, x)
                current_vertex = self.vertices[u]
                current_vertex.add_edge(edge)

    
    def climb(self, start, exits):
        """
        This function can choose to spend time to engage the monster and collect
        the key, or to not waste any time and travel along another path without collecting the
        key.
        This function would return one shortest route from start to one of the exits points that leads
        to the next floor. This route would need to include defeating one monster and collecting the key
        in order to go up to the next floor. Thus the function would return a tuple of (total_time,
        route):
            • total_time is the time taken to complete the route.
            • route is the shortest route as a list of integers that represent the location IDs along the
            path. If there are multiple routes that satisfy the constraints stated, return any one of
            those routes.
        If no such route exist, then the function would return None.

        written by Jessica Angeline Tjandra

        Precondition: start is a valid vertex id (0 <= start < len(self.vertices))
        Postcondition: shortest path from start to one of the exits is found

        Input:
            start: integer representing the start vertex id
            exits: list of integers representing the exit vertex ids
        Return:
            None if no such route exist
            (total_time, route) if such route exist
        
        Time complexity:
            Best == Worst: O(|E|log|V|), where E is the number of edges and V is the number of vertices.
        Space complexity:
            Input: O(|V|), where |V| is the number vertices.
            Aux: O(|V|), where |V| is the number of vertices.
        """
        # reset every time climb is called for different start and exits
        self.reset()
        route = []

        # find the shortest path from start to all vertices
        self.dijkstra(start)

        # make an inverted graph
        inverted_graph = FloorGraph(self.paths, self.keys, invert =True)
        dummy_node = inverted_graph.vertices[inverted_graph.max_vertex_id]
        dummy_edges = []
        for exit_ in exits:
            dummy_edges.append((exit_, dummy_node.id, 0)) #flip the order to make it inverted

        inverted_graph.add_edges(dummy_edges)

        # find the shortest path from the dummy node to all vertices
        inverted_graph.dijkstra(dummy_node.id)


        # find the key with the shortest total time to travel to
        total_time = float('inf')
        min_key = None

        # we only care about the vertices that have keys, therefore just loop through the keys list
        for key in self.keys:
            #                start to key + key to end                           + key's fight time
            if self.vertices[key[0]].time + inverted_graph.vertices[key[0]].time + key[1] < total_time:
                total_time = self.vertices[key[0]].time + inverted_graph.vertices[key[0]].time + key[1]
                min_key = key[0]

        # if there is no path from start to any of the exits with key, return None
        if total_time == float('inf'):
            return None
        
        # backtrack to get the route from key to start and key to exit
        path_start_to_key = self.backtrack(self.vertices[min_key])
        path_key_to_exit = inverted_graph.backtrack(inverted_graph.vertices[min_key])

        # combine the two paths
        route = path_start_to_key + path_key_to_exit

        
        return (total_time, route)
    
    def backtrack(self, vertex):
        """
        This function would return the shortest route from the start vertex to the given vertex
        as a list of integers that represent the location IDs along the path. If there are multiple
        routes that satisfy the constraints stated, return any one of those routes.
                
        written by Jessica Angeline Tjandra
        
        Precondition: vertex is a valid vertex id (0 <= vertex < len(self.vertices))
        Postcondition: path from start to vertex is found and stored in the returned list
        
        Input:
            vertex: integer representing the vertex id
        Return:
            route: list of integers representing the path from start to vertex
                
        Time complexity:
            Best == Worst: O(|V|), where |V| is the number of vertices.
        Space complexity:
            Input: O(1)
            Aux: O(|V|), where |V| is the number of vertices.
            
        """
        route = []
        current_vertex = vertex

        # keep going back to the previous vertices until it reaches 
        # the vertex with no previous vertex
        while current_vertex is not None:
            route.append(current_vertex.id)
            current_vertex = current_vertex.previous

        # no need to reverse the route if the graph is inverted
        if self.invert:
            return route[1:-1]
        return route[::-1]
    
    def dijkstra(self, start):
        """
        

        Approach: make a custom minheap of size N, each node stores the vertex_id and total time. plus an index array that keeps track of which index in the heap array, each element is in.
        in the first innitialisation of the heap, all time will be set to positive infinity.
        when the dijkstra finds a shorter time, it will use the MinHeap.update_shorter_time() that updates the total time value then adjust the position in the heap. the index array will also be adjusted accordingly.

        
        reference: FIT2004's Dijkstra's algorithm, https://github.com/sengweihan/Algorithm_And_DataStructure_FIT2004/blob/main/assignment4.py ,
        https://github.com/zhxnlee/FIT2004-Algorithms-and-data-structures/blob/main/Assignment2/Dijkstra%20algorithm.py
        
        written by Jessica Angeline Tjandra

        Precondition: start is a valid vertex id (0 <= start < len(self.vertices))
        Postcondition: shortest path from start to all vertices is found and stored in the vertices' time attribute

        Input:
            start: integer representing the start vertex id
        Return:
            None
        
        Time complexity:
            Best == Worst: O(|E|log|V|), where |E| is the number of edges and |V| is the number of vertices.
        Space complexity:
            Input: O(1)
            Aux: O(|V|), where |V| is the number of vertices in the heap.
        """
        priority_queue = MinHeap(self.max_vertex_id, start)  #list of [vertex id, total_time], O(V)
        self.vertices[start].added_to_queue() #vertex.discovered = True

        # if the priority queue (minheap) is not empty, meaning there are still vertices whose shortest time path is not finalized
        # hence, we want to find the shortest time path for all vertices so we need to make sure the minheap is empty
        # if the minheap is empty, meaning all vertices' shortest time path is finalized, because we only pop if and only if there
        # is no shorter path to that vertex
        while not priority_queue.is_empty():
            # get the vertex with the shortest time
            current_vertex = priority_queue.pop_min()  
            current_vertex_id = current_vertex[0]  
            current_vertex_time = current_vertex[1]    

            u = self.vertices[current_vertex_id]
            if u is None:
                break
            u.visit_node() #u.visited = True
            u.added_to_queue() #u.discovered = True

            # checks if there is no path from start to this vertex
            if current_vertex_time == float('inf'):
                u.time = float('inf')

            # count u's neighbours time taken to travel there
            # update the time of the neighbours in the heap and vertex.time attribute if it is shorter or just discovered it
            for edge in u.edges:
                neighbour_id = edge.v
                neighbour_time = edge.x

                v = self.vertices[neighbour_id] #get the vertex object

                # if the vertex is not in the heap [vertex.discovered == False], first time seeing it
                if not v.discovered:
                    v.added_to_queue() #v.discovered = True
                    v.time = u.time + neighbour_time
                    v.previous = u
                    priority_queue.update_shorter_time(neighbour_id, v.time)

                # if the vertex is in the heap [vertex.discovered == True] but the time is not finalized yet
                elif not v.visited:
                    if v.time > u.time + neighbour_time:
                        
                        v.time = u.time + neighbour_time
                        v.previous = u
                        priority_queue.update_shorter_time(neighbour_id, v.time)

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
        self.time = 0
        self.previous = None


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
    def __init__(self, u, v, x) -> None: 
        '''
        This function initialises the edge object.

        Precondition: u, v, x are valid values
        Postcondition: edge is initialised

        Input:
            u: integer representing the start vertex id
            v: integer representing the end vertex id
            x: integer representing the time taken to travel from u to v

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
        self.x = x #time

class MinHeap:
    """   Minheap class for the Dijkstra's algorithm    
    Only using |V| size of list, where |V| is the number of vertices in the graph.
    reference: FIT1008's Maxheap, https://github.com/jimmylu50/FIT2004/blob/main/Assignment4.py
    https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
    https://www.youtube.com/watch?v=8Q_B7vly1g4
    
    written by Jessica Angeline Tjandra
    """
    def __init__(self, length, start):
        '''
        This function initialises the heap array and index array.
        Written by Jessica Angeline Tjandra
        
        Precondition: length is a valid length of the heap array (length >= 0)
        Postcondition: heap array and index array are initialised
        
        Input:
            length: integer representing the length of the heap array
            
        Return:
            None
            
        Time complexity:
            Best == Worst: O(|V|), where |V| is the number of vertices/length/nodes of the heap.
        Space complexity:
            Input: O(1).
            Aux: O(|V|), where |V| is the number of vertices/length/nodes of the heap.
        '''
        if length < 0:
            raise ValueError("Length cannot be negative")

        self.heap_array = []
        self.index_array = []
        for vertex_id in range(length+1): # 0 - length
            self.heap_array.append([vertex_id, float('inf')]) # list of [vertex_id, total_time]
            self.index_array.append(vertex_id) #index_array[vertex_id] = index of vertex_id in heap_array
        
        # the time taken to go to itself is 0
        self.update_shorter_time(start, 0) #O(logV)

    def is_empty(self):
        """
        Function description: Check if the heap is empty.
        :Input: None
        :Output, return or postcondition: Boolean value representing the status of the heap is empty or not.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        # returns true if the heap is empty, false otherwise
        return len(self.heap_array) == 0
    
    def swap(self, index1, index2):
        '''
        This function swaps the elements at index1 and index2 in the heap array.
        Then updates the index array accordingly.
        Written by Jessica Angeline Tjandra

        Precondition: index1 and index2 are valid indices in the heap array (0 <= index1, index2 < len(self.heap_array)
        Postcondition: index array is updated according to the elements's indices in the heap array

        Input:
            index1: integer representing the index of the first element
            index2: integer representing the index of the second element

        Return:
            None

        Time complexity:
            Best == Worst: O(1)
        Space complexity:
            Input: O(1)
            Aux: O(1)
        '''
        # swaps item in heap_array
        self.heap_array[index1], self.heap_array[index2] = self.heap_array[index2], self.heap_array[index1]
        
        # update index_array
        self.index_array[self.heap_array[index1][0]] = index1
        self.index_array[self.heap_array[index2][0]] = index2

    def update_shorter_time(self, vertex_id, new_time):
        '''
        This function updates the time of the vertex in the heap with the new_time.
        Then rises the element at the proper place in the heap array sorting by the total time.
        then update the index_array
        Written by Jessica Angeline Tjandra

        Precondition: vertex_id is a valid vertex id (0 <= vertex_id < len(self.heap_array)
        Postcondition: minheap's property is maintained

        Input:
            vertex_id: integer representing the vertex id
            new_time: integer representing the new time
        Return:
            None

        Time complexity:
            Best == Worst: O(log|V|), where |V| is the number of vertices.
        Space complexity:
            Input: O(1)
            Aux: O(1)
        '''
        index = self.index_array[vertex_id]
        
        # update the time if index is valid
        if 0 <= index < len(self.heap_array):
            self.heap_array[index][1] = new_time
            self.rise(index) # O(log|V|)
        else:
            raise IndexError("Index out of range")

    def rise(self, index):
        '''
        This function rises the element to the proper position in the heap array sorting by the total time.
        Written by Jessica Angeline Tjandra, referenced from FIT1008's Maxheap

        Precondition: index is a valid index in the heap array (0 <= index < len(self.heap_array)
        Postcondition: minheap's property is maintained

        Input:
            index: integer representing the index of the element to be risen
        Return: 
            None
        
        Time complexity:
            Best == Worst: O(log|V|), where |V| is the number of vertices.

        Space complexity:
            Input: O(1)
            Aux: O(1)
        '''
        while index >= 0:
            parent_index = index // 2 
            # swaps if the current node is smaller than its parent
            if self.heap_array[index][1] < self.heap_array[parent_index][1]:
                self.swap(index, parent_index)
                index = parent_index
            else:
                break

    def sink(self, index):
        '''
        This function sinks the element at the proper place in the heap array sorting by the total time.
        Written by Jessica Angeline Tjandra

        Precondition: index is a valid index in the heap array (0 <= index < len(self.heap_array)
        Postcondition: minheap's property is maintained

        Input:
            index: integer representing the index of the element to be sunk
        Return:
            None

        Time complexity: 
            Best == Worst:  O(log|V|), where |V| is the number of vertices. 
        Space complexity: 
            Input: O(1)
            Aux: O(1)
        '''
        if index < 0 or index >= len(self.heap_array):
            raise IndexError("Index out of range")
        
        while True:
            # find the smallest item among the current node and its children
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index

            # check if the left child is smaller than the current smallest
            if (left_child_index < len(self.heap_array) and
                    self.heap_array[left_child_index][1] < self.heap_array[smallest][1]):
                smallest = left_child_index

            # check if the right child is smaller than the current smallest
            if (right_child_index < len(self.heap_array) and
                    self.heap_array[right_child_index][1] < self.heap_array[smallest][1]):
                smallest = right_child_index

            # swap the current node with the smallest if needed and update the index array
            # meaning the current node is not the smallest, so swap
            if smallest != index:
                self.swap(index, smallest)
                index = smallest

            else:
                break

    def pop_min(self):
        """
        This function pops and returns the vertex with the shortest time to travel to this vertex.
        It swaps the first and the last element in the heap array, then updates the index array.
        Written by Jessica Angeline Tjandra

        :Input: None
        :Output, return or postcondition: A list representing the vertex id and total time.
        :Time complexity: O(log|V|), where |V| is the number of vertices.
        :Aux space complexity:  O(1)
        """
        if self.is_empty():
            return None, None
        
        # get the minimum element
        min_item = self.heap_array[0]

        # swap the first and last element
        last_item = self.heap_array.pop()
        if not self.is_empty():
            self.heap_array[0] = last_item
            self.index_array[last_item[0]] = 0

            # adjust the proper place of this element in the heap
            self.sink(0)                        #O(logV)

        return min_item