from Libraries import *
from Tools import *
import pickle
def load_graphs(filenames):
    graphs = []
    for filename in filenames:
        with open(filename, "rb") as f:
            graph = pickle.load(f)
            graphs.append(graph)
    return graphs

def get_final_poses(graphs):
    list = []
    for graph in graphs:
        index = len(graph.pose)
        list.append(graph.pose[index-1])
    return list

def get_pose_diff(i_poses, f_poses):
    pose_diff = []
    for i_pose, f_pose in zip(i_poses, f_poses):
        diff = i_pose - f_pose
        diff[2] = angle_wrap(diff[2])  # Correct the angle difference
        pose_diff.append(diff)
    return pose_diff

def angle_wrap(angle_diff):
    """Wrap an angle difference to the range (-π, π]."""
    return (angle_diff + np.pi) % (2 * np.pi) - np.pi



def get_final_pose_change():
    f_graphs = load_graphs(filenames)
    i_graphs = load_graphs(init_filenames)

    f_poses = get_final_poses(f_graphs)
    i_poses = get_final_poses(i_graphs)

    pose_diff = get_pose_diff(i_poses, f_poses)

    # normed_poses = norm_pose(pose_diff)
    # for i in range(len(pose_diff)):
    #     print(filenames[i], pose_diff[i])

    return pose_diff

def norm_pose(pose_diff):
    normed_poses = []
    for pose in pose_diff:
        normed_pose = pose - pose_diff[4]
        normed_poses.append(normed_pose)
    for i in range(len(normed_poses)):
        print(filenames[i], normed_poses[i])
    return normed_poses
    
def get_errors(graphs, grountruths):#
    error_plot = []
    errors = []
    for graph, groundtruth in zip(graphs,groundtruths):
            error = get_pose_diff(graph.pose, groundtruth)
            
def get_edge_change(i_graphs, f_graphs):
    for i_graph, f_graph in zip(i_graphs, f_graphs):
        



filenames = ["0.1x_f.pkl", "0.1y_f.pkl", "0.1r_f.pkl", "0.1sens_f.pkl", "norm1_f.pkl", "10x_f.pkl", "10y_f.pkl", "10r_f.pkl", "10sens_f.pkl"]
init_filenames = ["0.1x_i.pkl", "0.1y_i.pkl", "0.1r_i.pkl", "0.1r_i.pkl", "norm1_i.pkl", "10x_i.pkl", "10y_i.pkl", "10r_i.pkl", "10sens_i.pkl"]
groundtruths = ["0.1x_grd.pkl", "0.1y_grd.pkl", "0.1r_grd.pkl", "0.1sens_grd.pkl", "norm1_grd.pkl", "10x_grd.pkl", "10y_grd.pkl", "10r_grd.pkl", "10sens_grd.pkl"]

f_graphs = load_graphs(filenames)
i_graphs = load_graphs(init_filenames)
grds = load_graphs(groundtruths)

print("pose)diff")
differential_poses = get_final_pose_change()
print("pose_norm")
norm_pose(differential_poses)
 

