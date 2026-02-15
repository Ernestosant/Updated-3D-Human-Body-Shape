import open3d as o3d
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / norm


def _apply_view(vis, lookat, front, up, zoom=0.7):
    ctrl = vis.get_view_control()
    ctrl.set_lookat(lookat.tolist())
    ctrl.set_front(_normalize(front).tolist())
    ctrl.set_up(_normalize(up).tolist())
    ctrl.set_zoom(float(zoom))


def _build_view_setters(vis, pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float64)

    def front_view(_):
        _apply_view(vis, center, np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        return False

    def side_view(_):
        _apply_view(vis, center, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        return False

    def top_view(_):
        _apply_view(vis, center, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
        return False

    def iso_view(_):
        _apply_view(vis, center, np.array([1.0, -1.0, 0.7]), np.array([0.0, 0.0, 1.0]))
        return False

    return front_view, side_view, top_view, iso_view

def visualize_depth(file_path):
    print(f"Loading: {file_path}")
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    try:
        # 1. Load Data
        depth_data = np.load(file_path).astype(np.float32)
        height, width = depth_data.shape
        
        # 2. Setup Camera Intrinsics
        fx, fy = 1000.0, 1000.0
        cx, cy = width / 2, height / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # 3. Create Open3D Geometry
        img_depth = o3d.geometry.Image(depth_data)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            img_depth, 
            intrinsic,
            depth_scale=1000.0, 
            depth_trunc=10000.0, 
            stride=1
        )
        
        points = np.asarray(pcd.points)
        print(f"Initial Points: {len(points)}")
        
        # FIX: Z-Inversion (restore convex shape)
        points[:, 2] = -points[:, 2]
        
        # --- SMART BACKGROUND REMOVAL ---
        # Strategy: The "background" (wall) is usually the largest cluster of points at a similar depth.
        # We compute a histogram of Z values.
        z_vals = points[:, 2]
        hist, bin_edges = np.histogram(z_vals, bins=100)
        
        # Find the bin with the most points (assumed to be the background wall)
        max_bin_idx = np.argmax(hist)
        wall_depth = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx+1]) / 2
        print(f"Detected Background Wall depth approx: {wall_depth}")
        
        # Filter out points near the wall depth (tolerance of e.g., 20cm)
        # Also filter extremely far/close outliers
        tolerance = 0.3 # 30cm tolerance zone around the wall
        
        # We define "Foreground" as anything DISTINCTLY closer or further than the wall.
        # Usually person is solid and separate.
        # Refined mask: Remove points within 'tolerance' of the wall_depth
        mask_background = np.abs(z_vals - wall_depth) < tolerance
        
        # Invert mask to keep foreground
        mask_foreground = ~mask_background
        
        # Apply mask
        pcd = pcd.select_by_index(np.where(mask_foreground)[0])
        print(f"Points after Wall Removal: {len(pcd.points)}")
        
        # Statistical outlier removal to clean up floating noise (hair/edges)
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        
        # --- JET COLORMAP ---
        points = np.asarray(pcd.points)
        if len(points) > 0:
            z_vals = points[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)
            
            # Matplotlib 'jet' colormap
            cmap = plt.get_cmap('jet')
            colors = cmap(z_norm)[:, :3] # RGBA -> RGB
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Center and Rotate
        pcd.translate(-pcd.get_center())
        R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R, center=(0, 0, 0))
        
        # Visualization
        print("Launching visualizer...")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Cleaned 3D View - {os.path.basename(file_path)}", width=1024, height=768, left=50, top=50)
        vis.add_geometry(pcd)

        front_view, side_view, top_view, iso_view = _build_view_setters(vis, pcd)
        vis.register_key_callback(ord('1'), front_view)
        vis.register_key_callback(ord('3'), side_view)
        vis.register_key_callback(ord('7'), top_view)
        vis.register_key_callback(ord('R'), iso_view)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05])
        opt.point_size = 4.0 # Nice big points

        print("Controles: mouse arrastrar=rotar, rueda=zoom, shift+arrastrar=pan")
        print("Vistas rápidas: 1=frontal, 3=lateral, 7=superior, R=isométrica/recentrar")

        iso_view(None)
        
        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to close...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_depth(sys.argv[1])
    else:
        print("Usage: python depth_visualizer.py <path_to_npy>")
