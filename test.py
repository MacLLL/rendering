import trimesh
import numpy as np

# Load the point cloud data
point_cloud = trimesh.load_mesh('path_to_point_cloud.ply')

# Set the desired voxel grid size
grid_shape = (128, 128, 128)

# Compute the voxel size based on the point cloud bounds and grid size
min_bound = point_cloud.bounds.min(axis=0)
max_bound = point_cloud.bounds.max(axis=0)
grid_size = max_bound - min_bound
voxel_size = grid_size / np.array(grid_shape)

# Create an empty voxel grid
voxel_grid = np.zeros(grid_shape, dtype=bool)

# Iterate over each point in the point cloud and set the corresponding voxel in the grid to True
for point in point_cloud.vertices:
    voxel_coordinates = ((point - min_bound) / voxel_size).astype(int)
    voxel_grid[tuple(voxel_coordinates)] = True
