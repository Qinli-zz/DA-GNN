# Directed-Acyclic-Graph-Neural-Network-for-Human-Motion-Prediction
Human motion prediction is essential in humanrobot interaction. Current research only considers the joint dependencies but ignores the bone dependencies and their relationship in the human skeleton, thus limiting the prediction accuracy. To address this issue, this paper considers the human skeleton as a directed acyclic graph with joints as vertexes and bones as directed edges. Then, a novel model named directed acyclic graph neural network (DA-GNN) is proposed. DAGNN follows the encoder-decoder structure. The encoder is stacked by multiple encoder blocks, each of which includes a directed acyclic graph computational operator (DA-GCO) to update joint and bone attributes based on the relationship between joint and bone dependencies in the observed human states, and a temporal update operator (TUO) to update the temporal dynamics of joints and bones in the same observation. After progressively implementing the above update process, the encoder outputs the final update result, which is fed into the decoder. The decoder includes a novel directed acyclic graph-based gated recurrent unit (DAG-GRU) and a multilayered perceptron (MLP), to predict future human states sequentially. To the best of our knowledge, it is the first time to consider the relationship between bone and joint dependencies in human motion prediction. Our experimental evaluations on two datasets, CMU Mocap and Human 3.6m, prove that DAGNN outperforms current models. Finally, we showcase the efficacy of DA-GNN in an realistic HRI scenario. 
# Module Requirement
Pytorch >= 3.6  
Numpy  
# Trainging
There are two codes for two different datasets: human3.6 and CMU Mocap. You could select one of them, e.g., human3.6

    cd code_for_human3.6

run

    python main.py
