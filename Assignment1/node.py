# This class represents a node for our list
class Node:
    def __init__(self, position:(), parent:()):
        #position is a tuple
        self.position = position
        #parent is another node
        self.parent = parent
        self.g = 0 # Distance to start node
        self.h = 0 # heuristic cost
        self.f = 0 # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    # Sort nodes by cost
    def __lt__(self, other):
         return self.f < other.f

    # Print node
    def __repr__(self):
        return '({0},{1})'.format(self.position, self.f)