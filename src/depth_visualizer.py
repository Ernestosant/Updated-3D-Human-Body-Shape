import open3d as o3d
import numpy as np
import sys
import os

def visualize_depth(file_path):
    print(f"Loading: {file_path}")
    
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    try:
        # 1. Load Data
        depth_data = np.load(file_path).astype(np.float32)
        height, width = depth_data.shape
        
        # 2. Setup Camera Intrinsics (Approximated)
        fx, fy = 1000.0, 1000.0
        cx, cy = width / 2, height / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # 3. Create Open3D Image
        img_depth = o3d.geometry.Image(depth_data)
        
        # 4. Generate Point Cloud
        # Using stride=1 for maximum quality since we are in a dedicated window
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            img_depth, 
            intrinsic,
            depth_scale=1000.0, 
            depth_trunc=10000.0,  # Increased to avoid cutting off data
            stride=1
        )
        
        # 5. Preprocessing & Background Removal
        print("Preprocessing point cloud...")
        points = np.asarray(pcd.points)
        
        # --- BACKGROUND REMOVAL STRATEGY ---
        # The user's image shows a clear person vs straight background.
        # We histogram the Z-values (depth) to find the "person" cluster.
        z_vals = points[:, 2]
        
        # Calculate histogram to find the background peak (usually the furthest or most common value)
        hist, bin_edges = np.histogram(z_vals, bins=100)
        
        # Assuming the person is closer than the background:
        # We look for a significant gap or threshold.
        # Simple heuristic: Keep points within a certain percentile range (eliminate far background)
        # Or simplistic: threshold max depth.
        
        # Let's try to remove points that are 'too far' relative to the median
        # The person is likely in the 'foreground'.
        # Note: We inverted Z earlier, so 'closer' might be larger or smaller depending on inversion.
        # Let's work with the raw depth distribution logic.
        
        # If we inverted Z: Z is negative. "Forward" is positive Z (if we faced camera).
        # Let's just look at the distribution.
        
        # Filter: keep only the "closest" 40% of points (assuming person fills < 40% of view)
        # or use standard deviation.
        valid_mask = np.abs(z_vals - np.mean(z_vals)) < 2.0 * np.std(z_vals)
        pcd = pcd.select_by_index(np.where(valid_mask)[0])
        
        # Refined statistical removal on the filtered cloud
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.5)
        
        # Recalculate points after filtering
        points = np.asarray(pcd.points)
        
        # Estimate Normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 1000.]))

        # Center the model
        pcd.translate(-pcd.get_center())
        
        # Rotate to be upright
        R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        pcd.rotate(R, center=(0, 0, 0))

        # 6. Point Cloud Visualization
        print("Preparing point cloud for visualization...")
        
        # --- JET COLORMAP (Heatmap style) ---
        # User showed a blue-to-yellow heatmap. Let's replicate that.
        # Map Z (depth) to color.
        # Since we rotated, Y is now up. But 'depth' is usually Z or original Z.
        # Let's map the current screen-Z (or whichever dimension represents depth now) to color.
        # After rotation (Pi around X), original Z became -Z (or Y became -Y).
        # Let's color by the new Z coordinate (depth relative to camera).
        
        curr_points = np.asarray(pcd.points)
        depth_vals = curr_points[:, 2] # Z axis
        
        # Normalize 0-1
        d_min, d_max = depth_vals.min(), depth_vals.max()
        if d_max - d_min > 0:
            d_norm = (depth_vals - d_min) / (d_max - d_min)
            
            # Simple Jet-like map: Blue -> Green -> Red
            colors = np.zeros_like(curr_points)
            
            # Blue to Cyan (0.0 - 0.33)
            mask1 = d_norm < 0.33
            colors[mask1, 0] = 0
            colors[mask1, 1] = d_norm[mask1] * 3
            colors[mask1, 2] = 1
            
            # Cyan to Yellow (0.33 - 0.66)
            mask2 = (d_norm >= 0.33) & (d_norm < 0.66)
            colors[mask2, 0] = (d_norm[mask2] - 0.33) * 3
            colors[mask2, 1] = 1
            colors[mask2, 2] = 1 - (d_norm[mask2] - 0.33) * 3
            
            # Yellow to Red (0.66 - 1.0)
            mask3 = d_norm >= 0.66
            colors[mask3, 0] = 1
            colors[mask3, 1] = 1 - (d_norm[mask3] - 0.66) * 3
            colors[mask3, 2] = 0
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        geometries = [pcd]
        window_name = f"3D Depth Visualization - {os.path.basename(file_path)} (Filtered)"

        # 7. Visualization
        print("Launching visualizer...")
        
        # Create visualization with custom render options
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1024, height=768, left=50, top=50)
        vis.add_geometry(pcd)
        
        # Increase point size for better visibility
        render_option = vis.get_render_option()
        render_option.point_size = 3.0  # Larger points
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to close...") # Keep window open to see error

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_depth(sys.argv[1])
    else:
        print("Usage: python depth_visualizer.py <path_to_npy>")
