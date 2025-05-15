from Libraries import *
from Tools import *
import pickle

with open("final_graph.pkl", "rb") as f:
    graph = pickle.load(f)
with open("init_graph.pkl", "rb") as f:
    init_graph = pickle.load(f)

with open("grdtruth.pkl", "rb") as f:
    gtruth = pickle.load(f)
# plot_graph_square(graph, gtruth)
plot_graph_square(graph, gtruth)
