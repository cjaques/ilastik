
import kdtree
import numpy as np

class KDTreeTransformer(object):
    
    def __init__(self, max_depth):
        
        self.kdtree = None
        self.max_depth = max_depth
    
    def fit(self, X, y=None):
        
        self.kdtree = kdtree.KDTree(self.max_depth)
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
    
