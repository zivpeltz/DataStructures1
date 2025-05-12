# username - zivpeltz
# id1      - 212936199
# name1    - Ziv Peltz
# id2      - 215975954
# name2    - Lior Bornstein


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


    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def is_real_node(self):
        return self.key is not None


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
        self.size = 0
        self.virtual = AVLNode(None, None)

    """searches for a node in the dictionary corresponding to the key

    @type key: int
    @param key: a key to be searched
    @rtype: AVLNode
    @returns: node corresponding to key
    """

    def search(self, key):
        return self.search_from_node(key, self.root)

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
        self.size += 1 #change size because we add a new node

        #Edge case: tree is empty
        if self.root == None:
            self.root = self.create_fresh_node(key,val)
            self.max = self.root
            return 0
        #Check from where to search and then commit search
        if start == "root":
            node = self.search_from_node(key, self.root)
        else:
            node = self.search_from_max(key)

        #creation of new node
        new_key_node = self.create_fresh_node(key, val)
        new_key_node.parent = node

        #decide which side son the new node should be
        if node.key < key:
            node.right = new_key_node
        else:
            node.left = new_key_node

        #change max point if necessary
        if(key > self.max.key or self.max == None):
            self.max = new_key_node

        countRotations = 0

        while node is not None:
            curr_BF = self.compute_BF(node)
            print(f"curr= {node.key}, BF = {curr_BF}")

            height_change = self.did_height_change(node)

            if abs(curr_BF) < 2 and not height_change:
                return countRotations

            elif abs(curr_BF) < 2 and height_change:
                self.update_height(node)
                node = node.parent
                continue

            else:
                self.update_height(node)
                if curr_BF == -2:
                    right_BF = self.compute_BF(node.right)
                    if right_BF == -1:
                        self.left_rotate(node)
                        countRotations += 1

                    elif right_BF == 1:
                        self.right_rotate(node.right)
                        self.left_rotate(node)
                        countRotations += 2

                elif curr_BF == 2:
                    left_BF = self.compute_BF(node.left)
                    if left_BF == -1:
                        self.left_rotate(node.left)
                        self.right_rotate(node)
                        countRotations += 2

                    elif left_BF == 1:
                        self.right_rotate(node)
                        countRotations += 1
        return countRotations

    def did_height_change(self, node):
        '''checks for changed height in node'''
        return node.height != 1 + max(node.left.height, node.right.height)

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
        while node.key > key:
            node = node.parent
        return self.search_from_node(key, node)

    def delete(self, node):
        return -1

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

    def size(self):
        return self.size

    def get_root(self):
        return self.root

    def get_amir_balance_factor(self):
        return "you still need to do this"



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
        node.height = 1 + max(node.left.height, node.right.height)






    def __repr__(self):
        def printree(root):
            if not root:
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


def main():
    print("=== AVL Tree Demo ===")
    tree = AVLTree()

    keys_to_insert = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15 ,16 ,17 ,80]
    print(f"Inserting keys: {keys_to_insert}\n")

    for key in keys_to_insert:
        print(f"\nInserting ({key}, 'val{key}'):")
        rotations = tree.insert(key, f"val{key}")
        print(f"Rotations performed: {rotations}")
        print("Current tree structure:")
        print(tree)

    print("\n=== Final Tree Structure ===")
    print(tree)

    print("\n=== Inorder Traversal (Sorted key-value pairs) ===")
    print(tree.avl_to_array())

    print("\n=== Root of the Tree ===")
    print(f"Root Key: {tree.get_root().key}, Value: {tree.get_root().value}")

    print("\n=== Size of the Tree ===")
    print(tree.size)
    print(tree.get_amir_balance_factor())

if __name__ == "__main__":
    main()
