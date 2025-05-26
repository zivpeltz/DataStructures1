# username - zivpeltz
# id1      - 212936199
# name1    - Ziv Peltz
# id2      - 215975954
# name2    - Lior Bornstein
import sys
import time
import matplotlib.pyplot as plt

"""A class representing a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @param key: key of your node
    @type value: string
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.bf = 0

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def is_real_node(self):
        return self.key is not None

    def is_leaf(self):
        return (not self.left.is_real_node()) and (not self.right.is_real_node())

    def children_count(self):
        counter = 0
        if self.left.is_real_node():
            counter += 1
        if self.right.is_real_node():
            counter += 1
        return counter


"""
A class implementing an AVL tree.
"""


class AVLTree(object):
    """
    Constructor, you are allowed to add more fields.
    """

    def __init__(self):
        self.root = None
        self.max = None
        self.num_of_nodes = 0
        self.bf_zero_count = 0
        self.virtual = AVLNode(None, None)
        self.is_BST = False



    """searches for a node in the dictionary corresponding to the key
    @type key: int
    @param key: a key to be searched
    @rtype: AVLNode
    @returns: node corresponding to key
    """

    def search(self, key):
        node = self.search_from_node(key, self.root)
        if node is None or node is self.virtual:
            return None
        if node.key == key:
            return node


    """inserts a new node into the dictionary with corresponding key and value

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: string
    @param val: the value of the item
    @param start: can be either "root" or "max"
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, key, val, start="root"):
        self.num_of_nodes += 1  # change size because we add a new node
        self.bf_zero_count += 1

        # Edge case: tree is empty
        if self.root is None or not self.root.is_real_node():
            self.root = self.create_fresh_node(key, val)
            self.max = self.root
            return 0
        # Check from where to search and then commit search
        if start == "root":
            node = self.search_from_node(key, self.root)
        else:
            node = self.search_from_max(key)

        # creation of new node
        new_key_node = self.create_fresh_node(key, val)
        new_key_node.parent = node

        # decide which side son the new node should be
        if node.key < key:
            node.right = new_key_node
        else:
            node.left = new_key_node

        # change max pointer if necessary
        if (key > self.max.key or self.max == None):
            self.max = new_key_node

        if self.is_BST:
            return

        countfixes = 0

        while node is not None:
            height_change = self.did_height_change(node)
            self.update_height(node)
            if height_change:
                countfixes += 1

            curr_BF = self.compute_BF(node)

            if abs(curr_BF) < 2 and not height_change:
                return countfixes

            elif abs(curr_BF) < 2 and height_change:
                node = node.parent
                continue

            else:
                countfixes -= 1  # offset for the last update height
                if curr_BF == -2:
                    right_BF = self.compute_BF(node.right)
                    if right_BF == -1:
                        self.left_rotate(node)
                        countfixes += 1

                    elif right_BF == 1:
                        self.right_rotate(node.right)
                        self.left_rotate(node)
                        countfixes += 2

                elif curr_BF == 2:
                    left_BF = self.compute_BF(node.left)
                    if left_BF == -1:
                        self.left_rotate(node.left)
                        self.right_rotate(node)
                        countfixes += 2
                        return countfixes


                    elif left_BF == 1:
                        self.right_rotate(node)
                        countfixes += 1
                        return countfixes

        return countfixes

    def did_height_change(self, node):
        '''checks for changed height in node'''
        return node.height != 1 + max(node.left.height, node.right.height)

    def update_bf_zero(self,node, old_bf):
        '''assumes height was changed'''
        if node is not None and not node.is_real_node():
            return
        curr_bf = self.compute_BF(node)
        if curr_bf != old_bf:
            if old_bf == 0:
                self.bf_zero_count -= 1
            if curr_bf == 0:
                self.bf_zero_count += 1

    def create_fresh_node(self, key, value):
        '''creates a new node object to insert'''
        node = AVLNode(key, value)
        node.right = self.virtual
        node.left = self.virtual
        node.height = 0
        return node

    def compute_BF(self, node):
        return node.left.height - node.right.height

    def search_from_node(self, key, node):
        """
        Returns:
          - If key exists: the node with node.key == key.
          - Otherwise: the parent (אב) under which the new key should be attached
            (i.e. where left or right is self.virtual).
        """
        # empty or virtual subtree → nothing to search here
        if node is None or node is self.virtual:
            return None

        current = node
        while True:
            if key == current.key:
                return current

            # go left subtree?
            if key < current.key:
                if current.left is self.virtual:
                    return current
                current = current.left

            # go right subtree
            else:
                if current.right is self.virtual:
                    return current
                current = current.right


    def search_from_max(self, key):
        """
        Starts at the max node and walks up until node.key <= key,
        then delegates to search_from_node.
        """
        node = self.max

        # if max isn’t set or we go past the root, use the true root
        while node is not None and node is not self.virtual and node.key > key:
            node = node.parent

        if node is None or node is self.virtual:
            node = self.root

        # now node is guaranteed real, hand off to the usual search
        return self.search_from_node(key, node)

    """deletes node from the dictionary

    @type node: AVLNode
    @pre: node is a real pointer to a node in self
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, node):
        node = self.delete_node(node)
        self.max = self.find_max()
        if self.is_BST:
            return

        if node is None : return 0
        countfixes = 0

        while node is not None:
            height_change = self.did_height_change(node)
            self.update_height(node)
            if height_change:
                countfixes += 1
            curr_BF = self.compute_BF(node)

            if abs(curr_BF) < 2 and not height_change:
                node = node.parent
                continue

            elif abs(curr_BF) < 2 and height_change:
                node = node.parent
                continue

            else:
                if curr_BF == -2:
                    right_BF = self.compute_BF(node.right)
                    if right_BF == -1 or right_BF == 0:
                        self.left_rotate(node)
                        countfixes += 1

                    elif right_BF == 1:
                        self.right_rotate(node.right)
                        self.left_rotate(node)
                        countfixes += 2

                elif curr_BF == 2:
                    left_BF = self.compute_BF(node.left)
                    if left_BF == -1:
                        self.left_rotate(node.left)
                        self.right_rotate(node)
                        countfixes += 2

                    elif left_BF == 1 or left_BF == 0:
                        self.right_rotate(node)
                        countfixes += 1
                node = node.parent
        return countfixes

    def delete_node(self, node):
        ''' deletes node as done in BST and returns pointer to the parent of the node'''
        if node.bf == 0:
            self.bf_zero_count -= 1

        self.num_of_nodes -= 1
        if node.is_leaf():  # case 1: node is leaf
            if node == self.root:  # edge case: tree is only root
                self.root = self.virtual
                return None

            parent = node.parent
            side = self.check_side(parent, node)
            if side == "left":
                parent.left = self.virtual
            else:
                parent.right = self.virtual
            return parent

        if node.children_count() == 1:  # case 2: one child
            if self.root == node:  # edge case: node is root
                if node.left == self.virtual:
                    self.root = node.right
                    node.right.parent = None
                else:
                    self.root = node.left
                    node.left.parent = None
                return None

            parent = node.parent
            side = self.check_side(parent, node)
            if node.left == self.virtual:
                son = node.right
            else:
                son = node.left

            self.bypass_node(parent, son, side)
            return parent

        # case 3: two children
        # find successor and separate it from the tree
        succ = self.find_successor(node)
        if succ.parent == node:
            # special case for optimization
            succ.left = node.left
            succ.parent = node.parent

            if node.left != self.virtual:
                node.left.parent = succ

            if node == self.root:
                self.root = succ
            else:
                side = self.check_side(node.parent, node)
                self.bypass_node(node.parent, succ, side)

            self.update_height(succ)
            return succ

        son = succ.right
        parent = succ.parent
        side = self.check_side(parent, succ)
        self.bypass_node(parent, son, side)
        self.replace_node(node, succ)
        if node == self.root:
            self.root = succ
        self.update_height(succ)
        return parent


    def find_max(self):
        """
        Returns the node with the largest key, or None if the tree is empty.
        """
        # empty tree or only virtual root?
        if self.root is None or not self.root.is_real_node():
            return None

        node = self.root
        # walk right until you hit a virtual (non-ריאלי) child
        while node.right is not None and node.right.is_real_node():
            node = node.right
        return node

    def check_side(self, parent, node):
        '''checks which side son of the given node is of a given parent'''
        if (parent.left == node):
            return "left"
        else:
            return "right"

    def replace_node(self, x, y):
        '''places a different node in the same place as another node'''
        y.right = x.right
        y.left = x.left
        y.parent = x.parent

        if x.left.is_real_node():
            x.left.parent = y
        if x.right.is_real_node():
            x.right.parent = y

        if self.root != x:
            side = self.check_side(x.parent, x)
            if side == "left":
                x.parent.left = y
            else:
                x.parent.right = y


    def bypass_node(self, parent, son, side):
        '''Replaces node x with node y in the tree structure.
        Assumes y is already disconnected from its original parent.'''
        if side == "left":

            parent.left = son
        else:
            parent.right = son


        if son.is_real_node():
            son.parent = parent


    """returns an array representing dictionary

    @rtype: list
    @returns: a sorted list according to key of touples (key, value) representing the data structure
    """

    def avl_to_array(self):
        return self.inorder_traversal()

    def inorder_traversal(self, node=None, result=None):
        '''returns array after doing an inOrder scan of the tree'''
        if result is None:
            result = []
        if node is None:
            node = self.root

        if node == self.virtual:
            return result

        self.inorder_traversal(node.left, result)
        result.append((node.key, node.value))
        self.inorder_traversal(node.right, result)
        return result

    """returns the number of items in dictionary 

    	@rtype: int
    	@returns: the number of items in dictionary 
    	"""

    def size(self):
        return self.num_of_nodes

    """returns the root of the tree representing the dictionary

    	@rtype: AVLNode
    	@returns: the root, None if the dictionary is empty
    	"""

    def get_root(self):
        return self.root

    """gets amir's suggestion of balance factor

       @returns: the number of nodes which have balance factor equals to 0 devided by the total number of nodes
       """

    def get_amir_balance_factor(self):
        if self.num_of_nodes == 0:
            return 0
        return self.bf_zero_count / self.num_of_nodes


    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left is not self.virtual:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        self.update_height(x)
        self.update_height(y)

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right is not self.virtual:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        self.update_height(x)
        self.update_height(y)


    def update_height(self, node):
        old_bf = node.bf
        node.height = 1 + max(node.left.height, node.right.height)
        node.bf = self.compute_BF(node)
        self.update_bf_zero(node,old_bf)

    def find_min(self, node):
        '''finds min node of tree or subtree'''
        while node.left.is_real_node():
            node = node.left
        return node

    def find_successor(self, node):
        '''finds successor of given node'''
        if (node.right.is_real_node()):
            return self.find_min(node.right)

        parent = node.parent

        while parent.is_real_node() and parent.right == node:
            node = parent
            parent = node.parent

        return parent





