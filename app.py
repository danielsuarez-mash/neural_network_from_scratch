import streamlit as st
import numpy as np
from load_data import DataLoader
from initialise_params import InitialiseParameters
from forward_prop import ForwardProp
from compute_cost import Cost
from backward_prop import BackwardProp
from update_params import ParameterUpdater

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
