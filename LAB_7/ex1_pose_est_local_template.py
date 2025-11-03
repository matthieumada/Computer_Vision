import open3d as o3d
import numpy as np
from tqdm import tqdm

# ICP method find the closest pooint to each point find the trnasformation 
# apply it find the closest point until we reachs the iteration or all points respect the threeshold

def load_pointclouds():
    """
    Load the object and scene point clouds from the dataset you want to use.
    """
    obj = o3d.io.read_point_cloud('./datasets/gnome_artificial/object-local.pcd')
    scn = o3d.io.read_point_cloud('./datasets/gnome_artificial/scene.pcd')
    return obj, scn

def show_pointclouds(obj, scn, window_name):
    """
    Display the object and scene point clouds in a visualizer window.

    Hint: Use `o3d.visualization.draw_geometries` to visualize.
    Link: https://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html
    """
    o3d.visualization.draw_geometries([obj, scn], window_name = window_name)
    return

def create_kdtree(scn):
    """
    Create a k-d tree from the scene point cloud for efficient nearest neighbor search.
    
    Expected input: Scene point cloud.
    Expected output: KDTreeFlann object.
    
    Hint: Use `o3d.geometry.KDTreeFlann`.
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html
    """
    tree = o3d.geometry.KDTreeFlann(scn)
    return tree

def set_ICP_parameters():
    """
    Set parameters for ICP.

    Expected output: Number of iterations (int) and squared distance threshold (float).
    """
    # iteration number 
    it = int(input("enter the number of iteration for ICP:"))

    # Squared distance threshold
    thressq = float(input("Enter the squared distance threshold for ICP:"))
    return it, thressq

def find_closest_points(obj_aligned, tree, thressq):
    corr = []
    """
    Find closest points between the object and the scene point clouds using the k-d tree.
    
    Expected input: Object point cloud, KDTreeFlann object, and squared distance threshold.
    Expected output: Correspondences as a Vector2iVector. Each element is a pair of indices: (object_index, scene_index).
    
    Hint: You need to go through each point of the aligned object and find the closest point in the scene.
    Hint: Use `tree.search_knn_vector_3d` to find the closest point and then apply the distance threshold (append the correspondence to output if within).
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html#open3d.geometry.KDTreeFlann.search_knn_vector_3d
    """
    for i, point in enumerate(obj_aligned.points):
        n, id, distance = tree.search_knn_vector_3d(point, 1)
        # selection of points within the threshold
        if distance[0] <= thressq:
            corr.append([i, id[0]])
    corr = np.array(corr)
    corr =o3d.utility.Vector2iVector(corr)
    return corr

def estimate_transformation(obj, scn, corr):
    """
    Estimate the transformation matrix from the correspondences between object and scene.
    
    Expected input: Object point cloud, scene point cloud, and correspondences.
    Expected output: Transformation matrix.
    
    Hint: Use `TransformationEstimationPointToPoint().compute_transformation`. You need to create an instance of the class first.
    Link: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.TransformationEstimationPointToPoint.html
    """
    # create an instance of the class
    estimate = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # compute the transformation matrix
    T = estimate.compute_transformation(obj, scn, corr)
    return T

def apply_pose(obj, T):
    """
    Apply the transformation matrix to the object point cloud.
    
    Expected input: Object point cloud and transformation matrix.
    Expected output: Transformed object point cloud.

    Hint: This is a very short function.  
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.transform
    """
    obj = obj.transform(T)
    return obj

def update_result_pose_ICP(pose, T):
    """
    Update the overall transformation (pose) by multiplying it with the current transformation.
    
    Expected input: Current pose (None on first iteration) and current transformation.
    Expected output: Updated pose.
    
    Hint: Very short if/else statement. Apply matrix multiplication (`@`) to update the pose.
    """
    if pose is None:
        return T.copy()
    if T is None:
        return pose
    return T @ pose 

def main():
    # Purpose align an object point cloud to a scene point cloud using ICP (Iterative Closest Point) algorithm.
    # Load object and scene point clouds
    obj, scn = load_pointclouds()
    obj.colors = o3d.utility.Vector3dVector(np.zeros_like(obj.points) + [255,0,0])

    # Show starting object and scene
    show_pointclouds(obj, scn, window_name='Before local alignment')
    o3d.io.write_point_cloud("./exo_1_result/exo1_Before_local_alignment.pcd", obj+ scn)

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
    o3d.io.write_point_cloud("./exo_1_result/After_local_alignment.pcd", obj + scn)

if __name__ == '__main__':
    main()
