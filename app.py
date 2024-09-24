import streamlit as st
import numpy as np
from load_data import DataLoader
from initialise_params import InitialiseParameters
from forward_prop import ForwardProp
from compute_cost import Cost
from backward_prop import BackwardProp
from update_params import ParameterUpdater

# nn visualisation packages
from pyvis.network import Network
import networkx as nx
import streamlit.components.v1 as components

st.write("# Neural network using numpy")

# load and prepare data
X_train, y_train, test = DataLoader("data/").prepare_data()

with st.sidebar:

    # select number of layers
    L = st.slider(label="#layers", min_value=1, max_value=10, step=1, value=3)

    # layer dims
    layer_dims = [784]

    # cycle through layers
    for l in np.arange(2, L):

        # select number of nodes
        nodes = st.number_input(label="#nodes in layer{}".format(l), value=20, step=1)

        # append to list
        layer_dims.append(nodes)

    # last layer
    output_nodes = st.number_input(
        label="#nodes in layer{}".format(L), value=10, step=1
    )
    layer_dims.append(output_nodes)

    # learning rate
    learning_rate = st.number_input(
        label="Learning rate", value=0.001, format="%f", step=0.001
    )

    # number of iterations
    num_iterations = st.number_input(label="Number of iterations", value=100, step=10)

# get parameters
parameters = InitialiseParameters(layer_dims).initialise_parameters()

if st.button(label="Train model"):

    st.write("Training starting...")

    # repeat for number of iterations
    for i in range(0, num_iterations):

        # forward propagation
        fp = ForwardProp(parameters)
        caches = fp.l_model_forward(X_train)
        AL = fp.AL

        # compute cost
        cost = Cost(AL, y_train).compute_cost()

        # back propagation
        bp = BackwardProp(AL, y_train, caches)
        grads = bp.l_model_backward()

        # update parameters
        pu = ParameterUpdater(parameters, grads, learning_rate)
        parameters = pu.update_parameters()

        # print cost
        if i % 10 == 0:
            st.write("Iteration", i, "- cost:", cost)

    st.write("Training complete. Final parameters: {}".format(parameters))

    # visualise network
    G = nx.DiGraph()

    x_distance = 500 # distance between layers
    base_y_distance = 300  # distance between nodes

    st.write(layer_dims)

    # add nodes
    for layer_no in range(1, len(layer_dims)): # cycle through layers
        
        # y_distance = base_y_distance/(layer_no + 1)
        st.write(layer_no)

        # get weights and biases for hidden layers
        if layer_no!=0:
            keys = list(parameters.keys())
            layer_weights = parameters[keys[layer_no-1]]
            layer_bias = parameters[keys[layer_no]]
        
        y_distance = base_y_distance * (1 - layer_no / (len(layer_dims)))
        st.write(y_distance)
        
        # print('layer:', layer_no)
        # print('y_distance:', y_distance)
        
        for node_no in range(layer_dims[layer_no]): # cycle through each node in each layer (max nodes is 100)
            
            # node_id format
            node_id = '{}_{}'.format(layer_no, node_no)
            
            # spacing
            x_pos = layer_no * x_distance
            # y_pos = node_no * y_distance
            y_pos = node_no * y_distance - (layer_dims[layer_no] - 1) * y_distance / 2

            st.write('node:', node_id)
            st.write('position', x_pos, y_pos)
            
            G.add_node(node_id, x=x_pos, y=y_pos, physics=False)

            # add title to node
            if layer_no!=0:
                G.nodes[node_id]['title']="Weight:" + str(layer_weights[node_no]) + "\n" + "Bias:" + str(layer_bias[node_no])
            else:
                G.nodes[node_id]['title']="Input node:" + str(node_no)
    # add edges
    for layer_no in range(1, len(layer_dims)): # cycle through layers
        for node_from in range(layer_dims[layer_no]): # cycle through each node in each layer
            if layer_no != (len(layer_dims)-1): # check if we are on the last layer
                for node_to in range(layer_dims[layer_no+1]): # cycle through edges for each node
        
                    G.add_edge('{}_{}'.format(layer_no, node_from), '{}_{}'.format(layer_no+1, node_to))

    # Create a Pyvis network
    net = Network(height='600px', width='700px', directed=True, notebook=True, cdn_resources='in_line')

    # import networkx graph
    net.from_nx(G)


    net.save_graph('streamlit_nn.html')
    HtmlFile = open('streamlit_nn.html', 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=435)