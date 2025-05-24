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
        if node.key == key:
            return node
        return None

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
        if self.root == None:
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
            if height_change:
                self.update_height(node)
                countfixes += 1

            curr_BF = self.compute_BF(node)

            if abs(curr_BF) < 2 and not height_change:
                return countfixes

            elif abs(curr_BF) < 2 and height_change:
                node = node.parent
                continue

            else:
                countfixes -= 1 #offset for the last update height
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
            - If key exists in the tree: the node with node.key == key.
            - Otherwise: the would-be parent under which to insert the new key
              (i.e., its left or right child is self.virtual).
        """
        # Start from given node
        current = node
        while current is not self.virtual:
            if key == current.key:
                return current

            if key < current.key:
                # If left is a virtual node, we stop here
                if current.left is self.virtual:
                    return current
                current = current.left
            else:  # key > current.key
                # If right is a virtual node, we stop here
                if current.right is self.virtual:
                    return current
                current = current.right

        # If we somehow started at a virtual root, just return None
        return None

    def search_from_max(self, key):
        '''commence search from max node'''
        node = self.max
        while node.key > key and node is not self.root:
            node = node.parent
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
            if height_change:
                self.update_height(node)
                countfixes += 1
            curr_BF = self.compute_BF(node)

            if abs(curr_BF) < 2 and not height_change:
                node = node.parent
                continue

            elif abs(curr_BF) < 2 and height_change:
                node = node.parent
                continue

            else:
                #countfixes -=1 #offset for the last update height
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
        node = self.root

        while node.right.is_real_node():
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

    def __repr__(self):
        def printree(root):
            if not root or root is self.virtual:
                return ["#"]

            root_key = str(root.key)
            left, right = printree(root.left), printree(root.right)

            lwid = len(left[-1])
            rwid = len(right[-1])
            rootwid = len(root_key)

            result = [(lwid + 1) * " " + root_key + (rwid + 1) * " "]

            ls = len(left[0].rstrip())
            rs = len(right[0]) - len(right[0].lstrip())
            result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

            for i in range(max(len(left), len(right))):
                row = ""
                if i < len(left):
                    row += left[i]
                else:
                    row += lwid * " "

                row += (rootwid + 2) * " "

                if i < len(right):
                    row += right[i]
                else:
                    row += rwid * " "

                result.append(row)

            return result

        return '\n'.join(printree(self.root))

    def _count_leaves_rec(self, node):
        """Recursively counts leaf (עלה) nodes in the subtree rooted at *node*."""
        if node is self.virtual:  # reached a virtual child
            return 0
        if node.is_leaf():  # both children are virtual ⇒ real leaf
            return 1
        # recurse on both sides
        return self._count_leaves_rec(node.left) + self._count_leaves_rec(node.right)

    # === public API ===
    def count_leaves(self):
        """
        @rtype: int
        @returns: the number of leaves (עלים) in the entire AVL tree (עץ).
        """
        if self.root is None or self.root is self.virtual:
            return 0
        return self._count_leaves_rec(self.root)

def main():
    main_with_plot()


def main_with_plot():
    lst_size = 200
    inversions = [x for x in range(200)]  # 0 to 200000 with step 1000
    insertion_times = []

    # Run the tests and collect timing data
    for inversion_count in inversions:
        arr = arr_maker(lst_size, True)
        arr = array_with_inversions(arr, inversion_count)
        tree = AVLTree()

        # Measure insertion time
        start_time = time.perf_counter()
        for j in range(lst_size):
            tree.insert(arr[j], "0", "root")
        tree.avl_to_array()  # Ensure the tree is balanced and converted to an array
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        insertion_times.append(elapsed_time)
        print(f"Number of inversions: {inversion_count}, Insertion time: {elapsed_time:.6f} seconds")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(inversions, insertion_times, linestyle='-', color='blue')

    # Add labels and title
    plt.xlabel('Number of Inversions')
    plt.ylabel('Insertion Time (seconds)')
    plt.title('Relationship Between Array Inversions and AVL Tree Insertion Time')

    # Add grid and improve layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

def experiment():
    sys.setrecursionlimit(10 ** 6)

    times = []  # Will become a 4x10 matrix
    options = [(True, "root"), (True, "max"), (False, "root"), (False, "max")]

    for BST_flag, insertFrom in options:
        test = []
        n = 2000
        for i in range(10):
            tree = AVLTree()
            tree.is_BST = BST_flag
            arr = arr_maker(n, False)
            t0 = time.perf_counter()
            for j in range(n):
                tree.insert(arr[j], "0", insertFrom)
            tree.avl_to_array()
            t1 = time.perf_counter()
            elapsed = t1 - t0
            test.append(elapsed)
            print(
                f"run: {i} , BST: {BST_flag}, length of array = {n}, sorted = False, insertFrom = {insertFrom}, time = {elapsed:.6f} sec")
            n += 2000
        times.append(test)

    # Output the 4x10 matrix
    for idx, row in enumerate(times):
        print(f"Case {idx + 1}: {row}")

    plot_results(times)

def plot_results(times):
    x_values = [i for i in range(10)]  # constant spacing for plotting
    actual_sizes = [2000*(i+1) for i in range(10)]  # real array sizes for labels
    labels = [
        "tree = BST, insertFrom='root'",
        "tree = BST, insertFrom='max'",
        "tree = AVL, insertFrom='root'",
        "tree = AVL, insertFrom='max'"
    ]
    for i, label in enumerate(labels):
        plt.plot(x_values, times[i], marker='o', label=label)

    plt.xlabel("Array Size (n)")
    plt.ylabel("Insertion Time (seconds)")
    # Use a logarithmic scale for the y-axis to better separate close lower values
    import numpy as np
    min_time = min([min(row) for row in times])
    max_time = max([max(row) for row in times])
    # Avoid log(0) by setting a small positive minimum if needed
    if min_time <= 0:
        min_time = min([min([y for y in row if y > 0] or [1e-8]) for row in times])
    plt.yscale("linear")
    # Set y-ticks at measured times for clarity, but only if they are unique and not too many
    all_times = sorted(set([round(y, 8) for row in times for y in row if y > 0]))
    if len(all_times) > 12:
        # Pick up to 12 linearly spaced ticks for clarity
        y_ticks = np.linspace(min_time, max_time, num=12)
        plt.yticks(y_ticks, [f"{v:.6g}" for v in y_ticks])
    else:
        plt.yticks(all_times, [str(v) for v in all_times])
    plt.title("reverse-sorted array complexity comparison between trees")
    plt.legend()
    plt.grid(True)
    plt.xticks(x_values, actual_sizes)  # set real sizes as tick labels
    plt.tight_layout()
    plt.show()

def arr_maker(length, sorted):
    if sorted:
        return [x + 1 for x in range(length)]
    return [x for x in range(length,0,-1)]


def array_with_inversions(arr, k):
    """
    Given a sorted array and a number of inversions k,
    returns a permutation of the array with exactly k inversions.
    """
    n = len(arr)
    result = arr.copy()

    # Check if k is within valid range
    max_inversions = n * (n - 1) // 2
    if k > max_inversions:
        raise ValueError(f"Cannot create more than {max_inversions} inversions for array of length {n}")

    # For k = max_inversions, simply reverse the array
    if k == max_inversions:
        return result[::-1]

    # Process elements from right to left
    i = n - 1
    while k > 0 and i > 0:
        # How many positions to the left can we move this element?
        positions = min(k, i)

        # Extract the value to move
        value = result[i]

        # Shift elements to the right to make room
        for j in range(i, i - positions, -1):
            result[j] = result[j - 1]

        # Insert the value at its new position
        result[i - positions] = value

        # Update k and move to the next element
        k -= positions
        i -= 1

    return result

if __name__ == "__main__":
    main()
