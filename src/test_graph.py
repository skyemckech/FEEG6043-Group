from Libraries import *
from Tools import *
import pickle

with open("final_graph.pkl", "rb") as f:
    graph = pickle.load(f)

with open("gaytruth.pkl", "rb") as f:
    gtruth = pickle.load(f)
plot_graph_square(graph, gtruth)

np.savetxt("output.csv", graph.H, delimiter=",", fmt='%d')