import numpy as np
import vtk
from vtk.util import numpy_support

class DepthMeshLoader:
    """Utility class to load .npy depth images and convert them to 3D meshes."""

    def __init__(self, x_scale=1.0, y_scale=1.0, z_scale=None, auto_scale=True, invert_depth=True):
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.auto_scale = auto_scale
        self.invert_depth = invert_depth

    def load_npy(self, filepath):
        """Loads a .npy file and returns the numpy array."""
        try:
            # Handle potential Python 2 pickle issues
            data = np.load(filepath, allow_pickle=True, encoding='latin1')
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def convert_to_mesh(self, depth_image, max_edge_dist=10.0, depth_min_thr=None, depth_max_thr=None, rotation=None):
        """
        Converts a depth image (2D numpy array) to vertices and facets.
        Optimized for VTK rendering with auto-scaling and noise removal.
        
        Args:
            depth_image: 2D numpy array of depth values.
            max_edge_dist: Max Z difference between adjacent pixels to create a facet.
            depth_min_thr: Minimum depth value to include.
            depth_max_thr: Maximum depth value to include.
            rotation: Function or matrix to apply to vertices.
        """
        h, w = depth_image.shape
        
        # Initial mask for non-zero values
        mask = depth_image > 0
        
        # Apply depth thresholds if provided
        if depth_min_thr is not None:
            mask &= (depth_image >= depth_min_thr)
        if depth_max_thr is not None:
            mask &= (depth_image <= depth_max_thr)
            
        if np.sum(mask) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        # Auto-calculate Z scale
        if self.auto_scale or self.z_scale is None:
            valid_depths = depth_image[mask]
            d_min = float(valid_depths.min())
            d_max = float(valid_depths.max())
            d_range = d_max - d_min
            
            if d_range > 0:
                target_range = max(w, h) * 0.5 
                self.z_scale = target_range / d_range
            else:
                self.z_scale = 1.0
        
        # Mapping from (v, u) to vertex index
        valid_indices = np.full((h, w), -1, dtype=int)
        valid_indices[mask] = np.arange(np.sum(mask))
        
        # Create vertices (X, Y, Z)
        v_coords, u_coords = np.mgrid[0:h, 0:w]
        x = u_coords[mask].astype(np.float32) * self.x_scale
        y = (h - v_coords[mask]).astype(np.float32) * self.y_scale
        
        depth_values = depth_image[mask].astype(np.float32)
        if self.invert_depth:
            z = (depth_image[mask].max() - depth_values) * self.z_scale
        else:
            z = depth_values * self.z_scale
        
        vertices = np.column_stack((x, y, z)).astype(np.float32)
        
        # Apply rotation if provided
        if rotation is not None:
            # rotation can be a 3x3 matrix
            vertices = vertices @ rotation.T
            
        # Centering
        center = np.mean(vertices, axis=0)
        vertices -= center
        
        # Create facets (triangles) with edge length check to remove "curtains"
        facets = []
        
        # Pre-calculate scaled depth image for distance checking
        # (Using the same logic as vertex creation but keeping it in image shape)
        if self.invert_depth:
            d_max_val = depth_image[mask].max()
            scaled_depth = (d_max_val - depth_image.astype(np.float32)) * self.z_scale
        else:
            scaled_depth = depth_image.astype(np.float32) * self.z_scale

        for v in range(h - 1):
            for u in range(w - 1):
                idx00 = valid_indices[v, u]
                idx01 = valid_indices[v, u + 1]
                idx10 = valid_indices[v + 1, u]
                idx11 = valid_indices[v + 1, u + 1]
                
                # Check for two triangles in the quad
                # Triangle 1: (v,u), (v+1,u), (v,u+1)
                if idx00 != -1 and idx10 != -1 and idx01 != -1:
                    # Check depth discontinuity (noise removal)
                    d00 = scaled_depth[v, u]
                    d10 = scaled_depth[v + 1, u]
                    d01 = scaled_depth[v, u + 1]
                    
                    if (abs(d00 - d10) < max_edge_dist and 
                        abs(d00 - d01) < max_edge_dist and 
                        abs(d10 - d01) < max_edge_dist):
                        facets.append([idx00, idx10, idx01])
                
                # Triangle 2: (v+1,u+1), (v,u+1), (v+1,u)
                if idx11 != -1 and idx01 != -1 and idx10 != -1:
                    d11 = scaled_depth[v + 1, u + 1]
                    d01 = scaled_depth[v, u + 1]
                    d10 = scaled_depth[v + 1, u]
                    
                    if (abs(d11 - d01) < max_edge_dist and 
                        abs(d11 - d10) < max_edge_dist and 
                        abs(d01 - d10) < max_edge_dist):
                        facets.append([idx11, idx01, idx10])
        
        return vertices, np.array(facets, dtype=np.int32)

    def create_vtk_polydata(self, vertices, facets):
        """Creates a vtkPolyData object from vertices and facets."""
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(vertices, deep=True))
        
        cells = vtk.vtkCellArray()
        # VTK facets need to be prepended with the number of points (3 for triangles)
        # facets_with_size = np.column_stack((np.full(len(facets), 3), facets)).flatten()
        # Optimized way to add cells:
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        
        # Construct cells array
        # VTK 9+ preferred method:
        connectivity = numpy_support.numpy_to_vtk(facets.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE)
        offsets = numpy_support.numpy_to_vtk(np.arange(0, len(facets) * 3 + 1, 3), deep=True, array_type=vtk.VTK_ID_TYPE)
        
        cells.SetData(offsets, connectivity)
        poly_data.SetPolys(cells)
        
        return poly_data
