
import numpy as np

def extended_margin(image, patch_shape, mode='constant', cval=0):
    
    ndims = len(patch_shape)
    new_shape = tuple(s+ps-1 for s, ps in zip(image.shape, patch_shape))
    new_shape = new_shape + image.shape[ndims:]
    inner_box = tuple(slice(ps//2, s+ps//2) for s, ps in zip(image.shape, patch_shape))
    
    extended = np.empty(new_shape, dtype=image.dtype)
    extended[:] = cval
    extended[inner_box] = image
    return extended

def grid_patches_centers(img_shape, patch_shape, step=1, full_size=False):
    """Return the centers of a grid of patches with given shape"""
    if not full_size:
        center_slices = tuple(slice(i//2, j - (i-i//2) + 1, step)
            for i,j in zip(patch_shape, img_shape))
    else:
        center_slices = tuple(slice(0, j, step)
            for j in img_shape)
    return np.reshape(np.mgrid[center_slices], (len(patch_shape),-1)).T

def get_patch(image, patch_shape, center):
    """Return a single patch with the given shape and center"""
    slices = tuple(slice(i-ps//2, i-ps//2+ps) for i,ps in zip(center, patch_shape))
    return image[slices]

def get_grid_patches(image, patch_shape, step=1, flat=True, full_size=False):
    """Return all the patches in a grid"""
    centers = grid_patches_centers(image.shape, patch_shape, step, full_size)
    return get_many_patches(image, patch_shape, centers, flat)

try:
    import _patch_utils
    import utils
    
    def _get_many_patches(ndims, image, patch_shape, centers, step=1):
        
        patch_shape = tuple(patch_shape)
        if len(patch_shape) != ndims:
            raise ValueError("patch_shape is not valid for %sd patches" % ndims)
        
        if image.ndim < ndims:
            raise ValueError("image is not valid for extracting %sd patches" % ndims)
        
        image = np.reshape(image, image.shape[:ndims]+(-1,))
        
        if ndims == 2:
            patches = _patch_utils._get_many_patches2d(image, patch_shape, centers, step)
        elif ndims == 3:
            patches = _patch_utils._get_many_patches3d(image, patch_shape, centers, step)
        else:
            raise NotImplementedError("fast patch extraction only implemented for 2D and 3D patches")
        
        return patches
    
    def set_many_patches(image, patch_shape, centers, values):
        patch_shape = tuple(patch_shape)
        centers = np.reshape(np.asarray(centers), (-1, len(patch_shape)))
        
        num_patches = len(centers)
        try:
            values = utils.broadcast_to_shape(values, (num_patches,) + patch_shape)
        except ValueError, e:
            values = np.reshape(values, (num_patches,) + patch_shape)
        
        if len(patch_shape) == 3:
            _patch_utils._add_many_patches3d(image, patch_shape, centers, values)
        else:
            raise NotImplementedError("set_many_patches works only for 3D images")
    
except ImportError, e:
    pass

def get_many_patches(image, patch_shape, centers,
                     flat=True, step=1, force_pure_python=False):
    """Return the patches at given centers"""
    
    patch_shape = tuple(patch_shape)
    centers = np.reshape(np.asarray(centers, dtype=np.int_), (-1, len(patch_shape)))
    
    ndims = len(patch_shape)
    if ndims in [2,3] and "_get_many_patches" in globals() and not force_pure_python:
        # 3d version (efficient Cython implementation)
        patches = _get_many_patches(ndims, image, patch_shape, centers, step)
    else:
        # Extract patches (pure Python version)
        grid_slices = tuple(slice(-(i//2), i-i//2, step) for i in patch_shape)
        grid = np.reshape(np.mgrid[grid_slices], (len(patch_shape), -1))
        points = tuple(np.int_(centers.T[:,:,np.newaxis]) + np.int_(grid[:,np.newaxis,:]))
        patches = image[points]
    
    # Compute the final patch shape taking into acount the step
    final_shape = tuple((sh - 1)/step + 1 for sh in patch_shape)
    
    channels = image.shape[len(patch_shape):]
    if not flat:
        patches = np.reshape(patches, (-1,) + tuple(final_shape) + channels)
    else:
        patches = np.reshape(patches, (len(patches), np.prod(final_shape + channels)))
    return patches
