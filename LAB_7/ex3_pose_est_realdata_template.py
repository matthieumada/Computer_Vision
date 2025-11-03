import open3d as o3d
import numpy as np
from tqdm import tqdm
import random

from ex1_pose_est_local_template import show_pointclouds, create_kdtree, set_ICP_parameters, find_closest_points, estimate_transformation, apply_pose, update_result_pose_ICP
from ex2_pose_est_global_template import set_RANSAC_parameters, compute_surface_normals, compute_shape_features, find_feature_matches, sample_3_random_correspondences, validate, update_result_pose

def load_pointclouds():
    """
    Load the real data set point clouds.
    The real data objects are stored as stl files, they need to be converted to point clouds.
    """
    print( "Conversion STL file to point cloud")
    obj_mesh = o3d.io.read_triangle_mesh("./datasets/parts_real/object1-global.stl")
    print( "Mesh loaded:", obj_mesh)
    scn = o3d.io.read_point_cloud('./datasets/parts_real/scene.pcd')
    print( "Scene point cloud loaded:", scn)

    # Sample points only on the mesh
    obj = obj_mesh.sample_points_uniformly(number_of_points=20000)
    print( "Object point cloud sampled from mesh:", obj)

    return obj, scn


def preprocess_pointclouds(obj, scn):
    """
    Preprocess the real data set point clouds.
    You should use some of the methods from last week.
    You should at LEAST downsample the point clouds using a voxel grid (Need to be same size).

    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    """
    voxel_size = 0.01  # adapt to the dataset
    obj = obj.voxel_down_sample(voxel_size)
    scn = scn.voxel_down_sample(voxel_size)

    search_param = o3d.geometry.KDTreeSearchParamKNN(30)
    obj.estimate_normals(search_param=search_param)
    scn.estimate_normals(search_param=search_param)

    return obj, scn

def RANSAC(obj, scn):
    """
    Implement the code from ex2_pose_est_global_solution.py here.
    """
    # Show starting object and scene
    # Load object and scene point clouds
    obj, scn = load_pointclouds()
    obj.colors = o3d.utility.Vector3dVector(np.zeros_like(obj.points) + [255,0,0])

    # Show starting object and scene
    show_pointclouds(obj, scn, window_name='Before global alignment')

    # Set RANSAC parameters
    it, thressq = set_RANSAC_parameters()

    # Compute surface normals
    compute_surface_normals(obj, scn)

    # Compute shape features
    obj_features, scn_features = compute_shape_features(obj, scn)

    obj_features = np.asarray(obj_features.data).T
    scn_features = np.asarray(scn_features.data).T

    # Find feature matches
    corr = find_feature_matches(obj_features, scn_features)

    # Create a k-d tree for scene
    tree = create_kdtree(scn)
    print("KD-Tree created",tree)
    # Start RANSAC
    random.seed(123456789)
    inliers_best = 0
    pose_best = None
    for i in tqdm(range(it), desc='RANSAC'):   
        # Sample 3 random correspondences
        corr_i = sample_3_random_correspondences(corr)
        
        # Estimate transformation
        T = estimate_transformation(obj, scn, corr_i)
        
        # Apply pose (to a copy of the object)
        obj_aligned = o3d.geometry.PointCloud(obj)
        obj_aligned = apply_pose(obj_aligned, T)
        
        # Validate
        inliers = validate(obj_aligned, tree, thressq)

        # Update result
        pose_best, inliers_best = update_result_pose(pose_best, T, inliers, inliers_best, obj_aligned)

    # Print pose
    print('Got the following pose:')
    print(pose_best)

    # Apply pose to the original object
    obj = apply_pose(obj, pose_best)

    # Show result
    show_pointclouds(obj, scn, window_name='After global alignment')

# Show result info
    print("Transformation:")
    print(result.transformation)
    print("Inlier RMSE:", result.inlier_rmse)
    print("Fitness:", result.fitness)

# Apply pose to object
    obj_ransac = obj_copy = obj.copy()
    obj_ransac.transform(result.transformation)

# Visualize
    o3d.visualization.draw_geometries([obj_ransac, scn], window_name="RANSAC Result")
    return obj, scn

def ICP(obj, scn):
    """
    Implement the code from ex1_pose_est_local_solution.py here.
    """
    # Purpose align an object point cloud to a scene point cloud using ICP (Iterative Closest Point) algorithm.
    # Load object and scene point clouds
    obj, scn = load_pointclouds()
    obj.colors = o3d.utility.Vector3dVector(np.zeros_like(obj.points) + [255,0,0])

    # Show starting object and scene
    show_pointclouds(obj, scn, window_name='Before local alignment')
    o3d.io.write_point_cloud("./exo_3_result/exo1_Before_local_alignment_object.pcd", [obj, scn])
    # Create a k-d tree for scene
    tree = create_kdtree(scn)
    print("KD-Tree created",tree)
    # Set ICP parameters
    it, thressq = set_ICP_parameters()

    # Start ICP
    pose = None
    obj_aligned = o3d.geometry.PointCloud(obj)
    for i in tqdm(range(it), desc='ICP'): # tqdm for progress bar 
        # 1) Find closest points
        corr = find_closest_points(obj_aligned, tree, thressq)
            
        # 2) Estimate transformation
        T = estimate_transformation(obj_aligned, scn, corr)
        
        # 3) Apply pose
        obj_aligned = apply_pose(obj_aligned, T)
        
        # 4) Update result
        pose = update_result_pose_ICP(pose, T)

    # Print pose
    print('Got the following pose:')
    print(pose)

    # Apply pose to the original object
    obj = apply_pose(obj, pose)

    # Show result object and scene
    show_pointclouds(obj, scn, window_name='After local alignment')
    o3d.io.write_point_cloud("./exo_3_result/After_local_alignment_object.pcd", [obj, scn])

    return obj, scn

def main():
    print("Loading point clouds")
    obj, scn = load_pointclouds()
    print("Preprocessing point clouds")
    obj, scn = preprocess_pointclouds(obj, scn)
    print("Starting RANSAC")
    obj, scn = RANSAC(obj, scn)
    print("Starting ICP")
    obj, scn = ICP(obj, scn)
    


if __name__ == '__main__':
    main()