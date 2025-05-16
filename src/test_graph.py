from Libraries import *
from Tools import *
import pickle
from matplotlib import pyplot as plt

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
    
def get_errors(graphs, groundtruths):#
    errors = []
    for graph, groundtruth in zip(graphs, groundtruths):
            error = get_pose_diff(graph.pose, groundtruth)
            errors.append(error)
    return errors

def get_landmark_errors(graphs):
    landmark_errors = []
    
    for graph in graphs:#
        errors = []
        for i, landmark_id in enumerate(graph.landmark_id_array):
            if landmark_id == 0:
                error = np.linalg.norm(graph.landmark[i] - [[2],[0]])
            if landmark_id == 1:
                error = np.linalg.norm(graph.landmark[i] - [[2],[2]])
            if landmark_id == 2:
                error = np.linalg.norm(graph.landmark[i] - [[0],[2]])
            if landmark_id == 3:
                error = np.linalg.norm(graph.landmark[i] - [[0],[0]])
            errors.append(error)
        landmark_errors.append(errors)
    return landmark_errors

    
def plot_that_shit(dataset1, dataset2, labels):            
    # Compute averages for Dataset 1
    avg_euclidean_distances = [
        np.mean([np.linalg.norm([vec[0], vec[1]]) for vec in inner_list]) for inner_list in dataset1
    ]
    avg_headings = [
        np.mean([vec[2] for vec in inner_list]) for inner_list in dataset1
    ]

    # Compute averages for Dataset 2
    avg_scalars = [
        np.mean(inner_list) for inner_list in dataset2
    ]

    # Plotting bar charts
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Bar chart for Dataset 1: Euclidean distance and heading
    bar_width = 0.35
    x = np.arange(len(labels))

    axes[0].bar(x - bar_width / 2, avg_euclidean_distances, bar_width, label='Avg. Euclidean Distance', color='skyblue')
    axes[0].bar(x + bar_width / 2, avg_headings, bar_width, label='Avg. Heading', color='salmon')
    axes[0].set_title("Dataset 1: Euclidean Distance and Heading")
    axes[0].set_ylabel("Average Value")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, axis='y')

    # Bar chart for Dataset 2: Scalar averages
    axes[1].bar(labels, avg_scalars, color='lightgreen')
    axes[1].set_title("Dataset 2: Scalar Averages")
    axes[1].set_ylabel("Average Value")
    axes[1].grid(True, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()

def bar_chart(dataset1, dataset2, labels):
    avg_euclidean_distances = [
    np.mean([np.linalg.norm([vec[0], vec[1]]) for vec in inner_list]) for inner_list in dataset1
]
    avg_headings = [
        np.abs(np.mean([vec[2] for vec in inner_list])) for inner_list in dataset1
    ]

    # Compute averages for Dataset 2
    avg_scalars = [
        np.mean(inner_list) for inner_list in dataset2
    ]

    # Plotting bar charts
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Bar chart for Euclidean distances
    axes[0].bar(labels, avg_euclidean_distances, color='skyblue')
    axes[0].set_title("Average pose distance error after loop closure")
    axes[0].set_ylabel("Distance error (m)")
    axes[0].grid(True, axis='y')
    axes[0].tick_params(labelbottom=False)

    # Bar chart for Headings
    axes[1].bar(labels, avg_headings, color='salmon')
    axes[1].set_title("Average heading error after loop closure")
    axes[1].set_ylabel("Heading error (rad)")
    axes[1].grid(True, axis='y')
    axes[1].tick_params(labelbottom=False)

    # Bar chart for Scalar Averages (Dataset 2)
    axes[2].bar(labels, avg_scalars, color='lightgreen')
    axes[2].set_title("Average landmark distance error after loop closure")
    axes[2].set_ylabel("Distance error(m)")
    axes[2].grid(True, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()

filenames = ["0.1x_f.pkl", "0.1y_f.pkl", "0.1r_f.pkl", "0.1sens_f.pkl", "norm1_f.pkl", "10x_f.pkl", "10y_f.pkl", "10r_f.pkl", "10sens_f.pkl"]
init_filenames = ["0.1x_i.pkl", "0.1y_i.pkl", "0.1r_i.pkl", "0.1r_i.pkl", "norm1_i.pkl", "10x_i.pkl", "10y_i.pkl", "10r_i.pkl", "10sens_i.pkl"]
groundtruths = ["0.1x_grd.pkl", "0.1y_grd.pkl", "0.1r_grd.pkl", "0.1sens_grd.pkl", "norm1_grd.pkl", "10x_grd.pkl", "10y_grd.pkl", "10r_grd.pkl", "10sens_grd.pkl"]
labels = ["0.1xNorthings","0.1xEastings","0.1xHeadings","0.1xSensor", "Control","10xNorthings","10xEastings","10xHeadings","10xSensor"]

f_graphs = load_graphs(filenames)
i_graphs = load_graphs(init_filenames)
grds = load_graphs(groundtruths)

errors = get_errors(f_graphs, grds)

landmark_errors = get_landmark_errors(f_graphs)
print(landmark_errors)
bar_chart(errors, landmark_errors, labels)
# bar_chart(errors, filenames)