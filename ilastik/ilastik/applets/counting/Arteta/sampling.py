import numpy as np

def random_coords_from_mask(num_samples, mask, random_state=None):
    
    if random_state is None:
        random_state = np.random.RandomState()
    
    coords = np.array(np.nonzero(mask)).T
    if num_samples is not None:
        indices = random_state.choice(coords.shape[0],
                                      num_samples,
                                      replace=False)
        coords = coords[indices]
    
    return tuple(coords.T)

def sample_images(images, num_samples, mask, random_state=None):
    
    coords = random_coords_from_mask(num_samples, mask, random_state)
    return tuple(i[coords] for i in images)

class ImageRandomSampler(object):
    
    def __init__(self, num_samples, mask, random_state=None):
        self.coords = random_coords_from_mask(num_samples, mask, random_state)
    
    def __call__(self, image):
        return image[self.coords]
    
