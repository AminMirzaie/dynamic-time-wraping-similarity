import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtw import *
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

def create_node_trace(G):
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for i, node in enumerate(G.nodes(data=True)):
        x, y = node[1]['pos']
        node_x.append(x)
        node_y.append(y)

        node_text.append(node[1]['text'])
        node_color.append(node[1]['color'])
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=16,
            line_width=0.5,
        ),
        text=node_text,
        visible=False
    )

    return node_trace
def create_edge_trace(G):
    # collect edges information from G to plot
    edge_weight = []
    edge_text = []
    edge_pos = []
    edge_color = []
    
    for edge in G.edges(data=True):        
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_pos.append([[x0, x1, None], [y0, y1, None]])
        edge_color.append("black")
    edge_traces = []
    for i in range(len(edge_pos)):        
        line_width = 1
        trace = go.Scatter(
            x=edge_pos[i][0], y=edge_pos[i][1],
            line=dict(width=line_width, color=edge_color[i]),
            mode='lines',
            visible=False
        )
        edge_traces.append(trace)

    return edge_traces
def filter_similarity_matrix_at_step(square_matrix, step_value):
    aux = square_matrix.copy()
    aux[aux <= step_value] = np.nan
    return aux
def get_interactive_slider_similarity_graph(square_matrix, slider_values, node_text=None, yaxisrange=None, xaxisrange=None):    
    fig = go.Figure()
    slider_dict = {}
    total_n_traces = 0
    for i, step_value in enumerate(slider_values):
        print(i)
        aux = filter_similarity_matrix_at_step(square_matrix, step_value)

        # create nx graph from sim matrix
        G = nx.to_networkx_graph(aux)
        
        # remove edges for 0 weight (NaN)
        G.remove_edges_from([(a, b) for a, b, attrs in G.edges(data=True) if np.isnan(attrs["weight"])])

        # assign node positions if None
        node_pos = nx.nx_pydot.graphviz_layout(G)

        # populate nodes with meta information
        for node in G.nodes(data=True):
            
            # node position
            node[1]['pos'] = node_pos[node[0]]

            # node color
            node[1]['color'] = "orange"

            # node text on hover if any is specified else is empty
            if node_text is not None:
                node[1]['text'] = node_text[node[0]]
            else:
                node[1]['text'] = ""
        # create edge taces (each edge is a trace, thus this is a list)
        edge_traces = create_edge_trace(G)
        # create node trace (a single trace for all nodes, thus it is not a list)
        node_trace = create_node_trace(G) 
        # store edge+node traces as single list for the current step value
        slider_dict[step_value] = edge_traces + [node_trace]
        # keep count of the total number of traces
        total_n_traces += len(slider_dict[step_value])
        # make sure that the first slider value is active for visualization
        if i == 0:
            for trace in slider_dict[step_value]:
                # make visible
                trace.visible = True

                
    # Create steps objects (one step per step_value)
    steps = []
    for step_value in slider_values:
        print(step_value)
        # count traces before adding new traces
        n_traces_before_adding_new = len(fig.data)
        # add new traces
        fig.add_traces(slider_dict[step_value])

        step = dict(
            # update figure when this step is active
            method="update",
            # make all traces invisible
            args=[{"visible": [False] * total_n_traces}],
            # label on the slider
            label=str(round(step_value, 3)),
        )

        # only toggle this step's traces visible, others remain invisible
        n_traces_for_step_value = len(slider_dict[step_value])
        for i in range(n_traces_before_adding_new, n_traces_before_adding_new + n_traces_for_step_value):
            step["args"][0]["visible"][i] = True
        
        # store step object in list of many steps
        steps.append(step)

    # create slider with list of step objects
    slider = [dict(
        active=0,
        steps=steps
    )]

    # add slider to figure and create layout
    fig.update_layout(
        sliders=slider,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(range=xaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=yaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        width=700, height=700,
    )

    return fig

# define slider steps (i.e., threshold values)
slider_steps = np.arange(0.4, 0.85, 0.05)
# get the slider figure

import pickle
with open('../data/index_province.pickle', 'rb') as handle:
    index_province= pickle.load(handle)
with open('../data/similarity.pickle', 'rb') as handle:
    distance_matrix = pickle.load(handle)
wordlist = index_province
fig = get_interactive_slider_similarity_graph(
    distance_matrix,
    slider_steps,
    node_text = list(wordlist.values())
)

print("here")
fig.show()