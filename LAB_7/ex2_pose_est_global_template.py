import open3d as o3d
import numpy as np
from tqdm import tqdm
import random

def load_pointclouds():
    """
    Load the object and scene point clouds from the dataset you want to use.
    """
    obj = o3d.io.read_point_cloud('./datasets/gnome_artificial/object-global.pcd')
    scn = o3d.io.read_point_cloud('./datasets/gnome_artificial/scene.pcd')
    return obj, scn

def show_pointclouds(obj, scn, window_name):
    """
    Display the object and scene point clouds in a visualizer window.

    Hint: Use `o3d.visualization.draw_geometries` to visualize.
    Link: https://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html
    """
    return

def set_RANSAC_parameters():
    """
    Set parameters for RANSAC.
    
    Expected input: None.
    Expected output: Number of iterations (int) and squared distance threshold (float).
    """
    return it, thressq

def compute_surface_normals(obj, scn):
    """
    Compute surface normals for both object and scene point clouds.
    
    Expected input: Object and scene point clouds.
    Expected output: None (normals are estimated in place, and saved in the point cloud object).
    
    Hint: Use `estimate_normals` with `KDTreeSearchParamKNN`.
    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html 
    (Under Vertex normal estimation is an example, but I recommend using search_param=o3d.geometry.KDTreeSearchParamKNN instead of HybridParam).
    """
    return

def compute_shape_features(obj, scn):
    """
    Compute shape features for both object and scene point clouds. The template assumes FPFH features.
    
    Expected input: Object and scene point clouds.
    Expected output: FPFH features for object and scene.
    
    Hint: Use `compute_fpfh_feature` with a `KDTreeSearchParamRadius`.
    Link: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.compute_fpfh_feature.html
    """
    return obj_features, scn_features

def find_feature_matches(obj_features, scn_features):
    """
    Find feature correspondences between the object and scene features.
    
    Expected input: Object and scene features.
    Expected output: Correspondences as a Vector2iVector. Each element is a pair of indices: (obj_feats_index, scene_feats_index).
    
    Hint: Loop over the object features, compute squared distances, and find the minimum.
    """
    return corr

def create_kdtree(scn):
    """
    Create a k-d tree from the scene point cloud for efficient nearest neighbor search.
    
    Expected input: Scene point cloud.
    Expected output: KDTreeFlann object.
    
    Hint: Use `o3d.geometry.KDTreeFlann`.
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html
    """
    return tree

def apply_pose(obj, T):
    """
    Apply the transformation matrix to the object point cloud.
    
    Expected input: Object point cloud and transformation matrix.
    Expected output: Transformed object point cloud.

    Hint: This is a very short function.  
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.transform
    """
    return obj

def sample_3_random_correspondences(corr):
    """
    Randomly sample 3 correspondences from the full correspondence set.
    
    Expected input: Correspondences as a Vector2iVector.
    Expected output: A set (Vector2iVector) of 3 random correspondences.
    
    Hint: Use `random.choices()` to sample 3 correspondences.
    Link: https://docs.python.org/3/library/random.html#random.choices
    """
    return random_corr

def estimate_transformation(obj, scn, corr):
    """
    Estimate the transformation matrix from the correspondences between object and scene.
    
    Expected input: Object point cloud, scene point cloud, and correspondences.
    Expected output: Transformation matrix.
    
    Hint: Use `TransformationEstimationPointToPoint().compute_transformation`. You need to create an instance of the class first.
    Link: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.TransformationEstimationPointToPoint.html
    """
    return T

def validate(obj_aligned, tree, thressq):
    """
    Validate the transformation by counting the number of inliers based on the distance threshold.
    
    Expected input: Object point cloud, KDTreeFlann object, and squared distance threshold.
    Expected output: Number of inliers (int).
    
    Hint: You need to go through each point of the aligned object and find the closest point in the scene.
    Hint: Use `tree.search_knn_vector_3d` to find the closest point and then apply the distance threshold.
          Then count the number of inliers (points where the closest point is within the threshold).
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html
    """
    return inliers

def update_result_pose(pose_best, T, inliers, inliers_best, obj):
    """
    Update the best pose based on the number of inliers found.
    
    Expected input: Best pose, current pose (T), number of inliers, best inliers, and object point cloud.
    Expected output: Updated pose and inlier count.
    
    Hint: Compare the current number of inliers to the best so far, and update pose_best and inliers_best they are greater.
    Hint: RANSAC, unlike ICP, does not accumulate transformations. Instead, you need to store the best transformation so far.
    Hint: The object point cloud "obj" is only here, so you can print something like this on success:
          print(f'Got a new model with {inliers}/{len(obj.points)} inliers!')
    """
    return pose_best, inliers_best

def main():
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
    
if __name__ == '__main__':
    main()