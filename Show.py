import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData

def load_points(file_path):
    """Loads 3D points from a PLY file."""
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return points

def plot_all_points_in_folders():
    """Plots all point files in folders matching Object*/points/ in the same plot with different colors."""
    base_pattern = 'Object*/points/*.ply'
    point_files = glob.glob(base_pattern)
    
    # Generate a list of colors
    cmap = plt.colormaps['hsv']
    colors = [cmap(i / len(point_files)) for i in range(len(point_files))]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, point_file in enumerate(point_files):
        points = load_points(point_file)
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        ax.scatter(x, y, z, c=[colors[idx]], marker='o', s=1, label=f'Point Cloud {idx+1}')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Point Clouds from Different Files')
    ax.legend()
    
    plt.show()

if __name__ == '__main__':
    plot_all_points_in_folders()
