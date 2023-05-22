import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

def save_image():
    # Load the PCD file
    pcd = o3d.io.read_point_cloud("top_view_pcd.pcd")

    # Extract the XYZ coordinates from the PCD
    points = np.asarray(pcd.points)

    # Get the minimum and maximum values in the X and Y axes
    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])

    # Define the resolution and scaling factor for the image
    resolution = 0.001  # Adjust as needed for desired image resolution
    scale = 1 / resolution

    # # Calculate the dimensions of the image
    width = int((max_x - min_x) * scale) + 1
    height = int((max_y - min_y) * scale) + 1

    # Create an empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert the 3D points to pixel coordinates
    x_pixels = ((points[:, 0] - min_x) * scale).astype(int)
    y_pixels = ((points[:, 1] - min_y) * scale).astype(int)

    # Set the color of each valid pixel in the image
    for x, y in zip(x_pixels, y_pixels):
        if 0 <= y < height and 0 <= x < width:
            image[y, x] = (255, 255, 255)  # Set pixel color to white

    # Plot and save the image
    plt.imshow(image)
    plt.axis("off")  # Remove axis labels
    plt.savefig("top_view.png", bbox_inches="tight", pad_inches=0)
    plt.show()

def deproject_depth_to_point_cloud(depth, fx, fy, cx, cy, scale, cam_pose):
    # Deproject depth values to a point cloud in 3D space
    # Inputs:
    # - depth: 2D array representing the depth values
    # - fx, fy: focal length in x and y directions
    # - cx, cy: principal point (optical center) in x and y directions
    # - scale: scaling factor for depth values
    # - cam_pose: camera pose transformation matrix
    # Returns:
    # - transformed_points: 3D points in the camera frame transformed by the camera pose

    # Get the dimensions of the depth map
    height, width = depth.shape

    # Create meshgrid of pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Compute 3D coordinates (x, y, z) from the depth values and camera intrinsics
    z = depth.flatten() * scale
    x = (u.flatten() - cx) * z / fx
    y = (v.flatten() - cy) * z / fy

    # Create homogeneous coordinates for the 3D points
    points_homogeneous = np.vstack((x, y, z, np.ones_like(x)))

    # Apply the camera pose transformation to the points
    transformed_points = np.matmul(cam_pose, points_homogeneous)
    transformed_points = transformed_points[:3, :].T

    return transformed_points


def top_view(new_pose, table_pcd):
    # Transform a point cloud to a top view given a new pose
    # Inputs:
    # - new_pose: new pose transformation matrix
    # - table_pcd: input point cloud
    # Returns:
    # - new_pcd: transformed point cloud in the top view

    # Extract the points from the input point cloud
    old_points = np.array(table_pcd.points)

    # Add a column of ones to the points for homogeneous coordinates
    ones_column = np.ones((old_points.shape[0], 1))
    old_points_with_ones = np.hstack((old_points, ones_column))

    # Apply the inverse of the new pose transformation to the points
    transformed_points = np.matmul(np.linalg.inv(new_pose), old_points_with_ones.T).T

    # Create a new point cloud and assign the transformed points
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])

    return new_pcd


if __name__ == '__main__':
    base_path = Path(__file__).parent.absolute()

    # Load depth map from file
    depth = np.load(base_path / 'depth.npy')

    # Camera Intrinsics
    fx, fy = 800, 800  # Focal lengths in x and y directions
    cx, cy = 640, 480  # Principal point (optical center) in x and y directions
    z_near, z_far = 0.05, 100.0  # Near and far clipping planes for depth values

    # Camera Extrinsics (pose transformation matrix)
    cam_pose = np.array([
        [0.0, -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],  # Row 1: x-axis orientation and translation
        [1.0, 0.0,           0.0,          0.0],  # Row 2: y-axis orientation and translation
        [0.0, np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],  # Row 3: z-axis orientation and translation
        [0.0, 0.0,           0.0,          1.0],  # Row 4: Homogeneous coordinates
    ])

    scale = 1  # Scaling factor for depth values
    point_cloud = deproject_depth_to_point_cloud(depth, fx, fy, cx, cy, scale, cam_pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])  # Visualize the original point cloud
    o3d.io.write_point_cloud("original_pcd.pcd", pcd)  # Save the top-down point cloud to a file

    voxel_size = 0.01  # Voxel size for downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size) # Downsampled the point cloud
    o3d.visualization.draw_geometries([downsampled_pcd])  # Visualize the downsampled point cloud

    # Segment the dominant plane in the point cloud
    plane_model, inliers = downsampled_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    # Select the table points by excluding the plane inliers
    table_pcd = downsampled_pcd.select_by_index(inliers, invert=False)
    o3d.visualization.draw_geometries([table_pcd])  # Visualize the objects point cloud

    # Select the table points by excluding the plane inliers
    object_pcd = downsampled_pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([object_pcd])  # Visualize the table point cloud

    # Currently, the PCD generated is Bottom-up view, defining a new pose for transforming the point cloud to a top-down view
    new_pose = np.eye(4)
    new_pose[1][1] = -1  # Flip the y-axis orientation
    new_pose[2][2] = -1  # Flip the z-axis orientation
    # new_pose[3][0] = 0.85  # Set the desired translation along the x-axis
    # new_pose[3][2] = 0.9  # Set the desired translation along the z-axis

    # Transform the table point cloud to a top-down view
    top_down_pcd = top_view(new_pose, object_pcd)
    o3d.visualization.draw_geometries([top_down_pcd])  # Visualize the top-down point cloud
    o3d.io.write_point_cloud("top_view_pcd.pcd", top_down_pcd)  # Save the top-down point cloud to a file

    # # Visualize the depth image...
    save_image()


