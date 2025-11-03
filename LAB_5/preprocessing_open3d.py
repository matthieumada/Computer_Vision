import open3d as o3d
import numpy as np
from tqdm import tqdm

# This function just displays the effect of one of the functions visually, feel free to ignore or remove it.
def display_removal(preserved_points, removed_points):
    removed_points.paint_uniform_color([1, 0, 0])        # Show removed points in red
    preserved_points.paint_uniform_color([0.8, 0.8, 0.8])# Show preserved points in gray
    o3d.visualization.draw_geometries([removed_points, preserved_points])

def voxel_grid(input_cloud):
    # Downsample the pointcloud using a Voxel Grid Filter.
    downsample = 10**(-6)# 1mm voxel grid 
    output_cloud = o3d.geometry.PointCloud.voxel_down_sample(input_cloud, voxel_size=downsample)
    return output_cloud

def outlier_removal(input_cloud):
    # Remove outliers from the pointcloud using a statistical outlier removal filter

    # Hint: The filter will return an index variable "ind", that can be used to select the PC, like so:
    #
    cl, ind = input_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio = 2.0)
    # number of neighbors to analyze for each point and standart deviation ratio 

    display_removal(input_cloud.select_by_index(ind), input_cloud.select_by_index(ind, invert=True))
    return input_cloud.select_by_index(ind)

def spatial_filter(input_cloud):
    # Remove points in the pointcloud that do not belong to the bust, using a Spatial Filter (passthrough)
    
    # Hint: You can implement this by simply cropping the pointcloud within min/max bounds
    output_cloud = o3d.geometry.crop_point_cloud(input_cloud, min_bound = (-0.1, -0.1, -0.1), max_bound = (0.1, 0.1, 0.1))
    return output_cloud 

def main(): ### Try other parameters for the filters to see how they affcet the results ###
    # Load pointcloud (unfiltered)
    cloud = o3d.io.read_point_cloud('/home/delinm/Documents/Robotics_Computer_Vision/Computer_Vision/LAB_5/cloud.pcd')

    # Show
    o3d.visualization.draw_geometries([cloud], window_name = 'Pointcloud before filtering')

    #print("PointCloud before filtering: {} data points".format(cloud.points.shape[0]))
    print("PointCloud before filtering: {} data points".format(len(cloud.points)))

    cloud_filtered = voxel_grid(cloud)
    #cloud_filtered = outlier_removal(cloud)
    #cloud_filtered = spatial_filter(cloud_filtered)

    #print("PointCloud after filtering: {} data points".format(cloud_filtered.points.shape[0]))
    print("PointCloud after filtering: {} data points".format(len(cloud_filtered.points)))

    o3d.io.write_point_cloud("cloud_voxel_py.pcd", cloud_filtered)

    # Show
    o3d.visualization.draw_geometries([cloud_filtered], window_name = 'Pointcloud after filtering')

if __name__ == "__main__":
    main()
