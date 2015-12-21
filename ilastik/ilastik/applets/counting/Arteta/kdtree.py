
import numpy as np

# TODO. To be renamed
class MyTree(object):
    
    def __init__(self, max_depth=None):
        self.tree_dtype = np.dtype([
                        ('dim', np.int32), # If < 0, it is a leaf.
                        ('value', np.float_), # If leaf, the estimation. If not leaf, the threshold.
                        ('score', np.float_),
                        ('left', np.int32), # If leaf, the estimation. If not leaf, the threshold.
                        ('right', np.int32) # If leaf, the estimation. If not leaf, the threshold.
                    ])
        
        self.max_depth = max_depth
    
    def fit(self, x, weights):
        
        self.tree = []
        self.data = np.copy(x)
        self.weights = np.copy(weights)
        
        # import pdb
        # pdb.set_trace()
        
        self._recursive_build_tree(self.data, self.weights, 0)
        
        self.tree = np.array(self.tree)
        
        return self
    
    def _recursive_build_tree(self, data, weights, depth):
        
        num_features = data.shape[-1]
        
        score = max(weights.sum(), 0)
        
        splits = [self._split_score(data, weights, i) + (i,) for i in xrange(num_features)]
        best_split = max(splits, key=lambda x: x[0])
        
        split_score, split_threshold, split_order, split_argmax, split_dimension = best_split
        
        left_order = split_order[:split_argmax]
        right_order = split_order[split_argmax:]
        if (score >= split_score or len(left_order) == 0 or len(right_order) == 0) \
                or (self.max_depth is not None and depth >= self.max_depth):
            # Leaf node
            new_node = np.array((-1, score > 0, score, -1, -1), dtype=self.tree_dtype)
            self.tree.append(new_node)
            node_index = len(self.tree) - 1
        else:
            # Split node
            new_node = np.array((split_dimension, split_threshold, score, -2, -2), dtype=self.tree_dtype)
            self.tree.append(new_node)
            node_index = len(self.tree) - 1
            
            # Recursively create children nodes
            node_left = self._recursive_build_tree(data[left_order], weights[left_order], depth+1)
            node_right = self._recursive_build_tree(data[right_order], weights[right_order], depth+1)
            new_node['left'] = node_left
            new_node['right'] = node_right
        
        return node_index
    
    def _split_score(self, data, weights, split_dimension):
        
        order = data[:, split_dimension].argsort()
        
        weights_cumsum = np.hstack([0, weights[order].cumsum()])
        weights_sum = weights_cumsum[-1]
        
        argmax1 = weights_cumsum.argmax()
        argmax2 = (weights_sum - weights_cumsum).argmax()
        
        if weights_cumsum[argmax1] > (weights_sum - weights_cumsum[argmax2]):
            argmax = argmax1
            score = weights_cumsum[argmax]
        else:
            argmax = argmax2
            score = weights_sum - weights_cumsum[argmax]
        
        if argmax == len(order):
            threshold = data[order[-1], split_dimension] + 10
        else:
            threshold = data[order[argmax], split_dimension]
        
        return score, threshold, order, argmax
    
    def get_weights(self, data, roots=None):
        
        tree = self.tree
        data = np.asarray(data)
        
        if roots is None:
            roots = np.zeros(len(data), dtype=np.int32)
        
        res = np.empty(len(data), dtype=np.float_)
        
        dims = tree['dim'][roots]
        values = tree['value'][roots]
        
        # Detect leaves
        leaf_mask = dims < 0
        notleaf_mask = np.logical_not(leaf_mask)
        
        res[leaf_mask] = values[leaf_mask]
        
        if np.any(notleaf_mask):
            data = data[notleaf_mask]
            roots = roots[notleaf_mask]
            
            dims = dims[notleaf_mask]
            values = tree['value'][roots]
            lefts = tree['left'][roots]
            rights = tree['right'][roots]
            
            split = data[np.arange(len(data)), dims] < values
            new_roots = np.where(split, lefts, rights)
            
            res[notleaf_mask] = self.get_weights(data, new_roots)
        
        return res
    

class KDTree(object):
    
    def __init__(self, max_depth=1, keep_data=False):
        self.max_depth = max_depth
        self.num_nodes = 2**self.max_depth - 1
        self.keep_data = keep_data
    
    def fit(self, data):
        self.tree_dtype = np.dtype([
                        ('dim', np.int32), # If < 0, it is a leaf
                        ('thresh', data.dtype),
                        ('start', np.int32),
                        ('end', np.int32)
                    ])
        
        self.tree = np.zeros(self.num_nodes, dtype=self.tree_dtype)
        
        data = np.copy(data)
        self._recursive_build_tree(0, data, 0)
        
        if self.keep_data:
            self.data = data
    
    def _recursive_build_tree(self, index, data, start):
        
        tree = self.tree
        
        tree[index]['start'] = start
        tree[index]['end'] = start + len(data)
        if 2*index+1 >= self.num_nodes:
            last_leaf = tree['dim'].min()
            tree[index]['dim'] = last_leaf - 1
            return
        
        # Find the dimension of maximum variance.
        dim = np.argmax(np.var(data, axis=0))
        median = np.median(data[:,dim])
        
        data[:] = data[data[:, dim].argsort()]
        
        half = len(data) / 2
        tree[index]['dim'] = dim
        tree[index]['thresh'] = median
        self._recursive_build_tree(2*index + 1, data[:-half], start)
        self._recursive_build_tree(2*index + 2, data[-half:], start+len(data)-half)
    
    def get_num_leaves(self):
        
        # return -self.tree['dim'].min()
        return 2**(self.max_depth - 1)
    
    def get_leaves(self, data, roots=None):
        
        tree = self.tree
        data = np.asarray(data)
        
        if roots is None:
            roots = np.zeros(len(data), dtype=np.int32)
        
        res = np.empty(len(data), dtype=np.int32)
        
        dims = tree['dim'][roots]
        
        # Detect leaves
        leaf_mask = dims < 0
        notleaf_mask = np.logical_not(leaf_mask)
        
        res[leaf_mask] = - dims[leaf_mask] - 1
        
        if np.any(notleaf_mask):
            data = data[notleaf_mask]
            roots = roots[notleaf_mask]
            
            dims = dims[notleaf_mask]
            thresholds = tree['thresh'][roots]
            
            split = data[np.arange(len(data)), dims] <= thresholds
            new_roots = np.where(split, 2*roots+1, 2*roots+2)
            
            res[notleaf_mask] = self.get_leaves(data, new_roots)

        return res


class KDTreeTransformer(object):
    
    def __init__(self, max_depth):
        
        self.kdtree = None
        self.max_depth = max_depth
    
    def fit(self, X, y=None):
        
        self.kdtree = KDTree(self.max_depth)
        self.kdtree.fit(X)
        
        return self
    
    def fit_transform(self, X, y):
        
        self.fit(X, y)
        return self.transform(X)
    
    def transform(self, X):
        
        leaves = self.kdtree.get_leaves(X)
        num_leaves = self.kdtree.get_num_leaves()
        
        res = np.zeros((len(X), num_leaves), dtype=np.float32)
        res[np.arange(len(X)), leaves] = 1
        return res
    
    def get_output_ndims(self):
        return self.kdtree.get_num_leaves()
    
    def get_params(self):
        
        return {"max_depth": self.max_depth}
    

    
