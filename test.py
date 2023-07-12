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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot the voxel grid
fig = plt.figure()
ax = fig.gca(projection='3d')

# Get the coordinates of the occupied voxels
x, y, z = voxel_grid.nonzero()

# Plot the occupied voxels
ax.voxels(voxel_grid, facecolors='b', edgecolor='k')

# Set the aspect ratio and labels
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust the plot limits
ax.set_xlim(0, voxel_grid.shape[0])
ax.set_ylim(0, voxel_grid.shape[1])
ax.set_zlim(0, voxel_grid.shape[2])

# Show the plot
plt.show()




import numpy as np

def point_cloud_to_voxel(point_cloud, voxel_size, grid_shape):
    # Compute the minimum and maximum coordinates of the point cloud
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)
    
    # Compute the voxel grid dimensions
    grid_size = max_bound - min_bound
    voxel_grid = np.zeros(grid_shape, dtype=bool)
    
    # Calculate the voxel indices for each point in the point cloud
    indices = ((point_cloud - min_bound) / voxel_size).astype(int)
    
    # Set the corresponding voxel grid values to True
    voxel_grid[tuple(indices.T)] = True
    
    return voxel_grid

# Sample point cloud data
point_cloud = np.array([[1.2, 3.4, 2.1],
                       [0.5, 2.3, 1.7],
                       [1.8, 1.2, 0.6],
                       [0.9, 3.6, 2.8]])

# Voxel parameters
voxel_size = 0.1
grid_shape = (10, 10, 10)

# Convert point cloud to voxel grid
voxel_grid = point_cloud_to_voxel(point_cloud, voxel_size, grid_shape)

# Print the voxel grid
print(voxel_grid)
