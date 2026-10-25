# COMP3710 Project: Multi-layer Graph Neural Network for Classifying Facebook Pages

## The Open Research Question
In web-scale graph classification, models frequently achieve deceptively high aggregate
test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, 
or out-of-distribution cases. This can be problematic for automatic classification, as 
incorrect predictions may not be identified for further review because the model is 
extremely confident in its classification. The intended users who would benefit from
automatic classification are social network analysts seeking to reduce manual 
classification, so it is important that the model works effectively.

This project will investigate whether a multi-layer Graph Neural Network (GNN),
that is, a two-layer Graph Convolutional Network (GCN), can produce a more reliable 
confidence score than a Multi-Layer Perceptron (MLP) baseline when classifying facebook 
pages.

A confidence threshold will be investigated to determine whether uncertain predictions
can be sent for manual review, in order to reduce the risk of automatically accepting
incorrect classifications. The results will then be used to evaluate whether the
additional complexity of the GCN and computational costs justify its performance
when compared to the simpler MLP baseline.

## Background Research and GNN Model Selection
Graph-based classification can benefit from using both the features of individual nodes 
and the connections between them. Rozemberczki et al. [1] investigated attributed node 
embeddings, where information from a node and its neighbouring nodes is used to learn 
useful representations. By incorporating information from neighbouring nodes (nodes that 
are connected through edges in the graph), their research investigated whether patterns in 
the surrounding network could provide additional information that improves node 
classification.

Rozemberczki et al. [1] evaluated several graph-based learning methods
using the Facebook Large Page-Page Network. Their results showed that a Graph 
Convolutional Network (GCN) achieved a test micro-F1 score of 0.932, compared with 0.919 
for a Graph Attention Network (GAT) and 0.941 for MixHop. Although MixHop achieved a 
higher score, the GCN was chosen for this project because it provides a simpler 
architecture while still demonstrating strong classification performance. This makes it 
suitable for investigating whether incorporating graph connections improves classification 
performance and confidence reliability when compared with an MLP baseline.

The setup used by the paper used a two-layer GCN with 32 hidden filters, the Adam 
optimiser, a learning rate of 0.01, 200 training epochs, an 80/20 training-test split, 
and dropout of 0.5. These parameter and training settings will be used as a good starting
point for the GCN model in this project, however a validation set is going to be used
as well as training and test to determine the best model.

The research primarily evaluates classification performance and does not investigate 
whether the models produce reliable confidence scores when their predictions are 
incorrect. This is the main area investigated in this project, where the GCN will be 
compared against an MLP baseline to determine whether incorporating graph connections 
improves classification performance and confidence reliability. A confidence threshold 
will also be evaluated to investigate whether uncertain predictions can be identified 
for manual review.

## The Proposed Solution

## How the Algorithm Works

# Dependencies and Reproducability
## Software Dependencies and Versions
This project is implemented in Python using PyTorch and PyTorch Geometric.
PyTorch is used to implement and train the MLP and GCN models, while PyTorch 
Geometric provides the Facebook Page-Page dataset and the GCNConv layers used 
by the GCN.

Due to issues in the PyTorch Geometric dataset, the Facebook Page-Page dataset
was obtained from this link:

https://github.com/gayanku/facebook_page-page_dataset/tree/main

The libraries and their version used for this project are:
```
Python               3.10.12
PyTorch:             2.14.1+cu130
PyTorch Geometric:   2.8.0.post1
NumPy:               2.2.6
scikit-learn:        1.7.2
Matplotlib:          3.10.9
CUDA:                13.0
```
The project also uses standard Python libraries including argparse, csv and time. 
These are included with Python and do not require separate installation.

## Hardware and Computing Environment
The project is developed locally using Ubuntu through Windows Subsystem for Linux 
(WSL). A Python virtual environment named gnn_env is used to manage the project
dependencies.

Model training and evaluation can be performed using either a CPU or a CUDA-compatible
GPU, depending on availability. The training script automatically selects CUDA when 
supported, otherwise it uses the CPU.

## Project File Structure
The project is contained in the recognition/MultiLayer_GNN_s4801265/ directory and
consists of the five mandatory files:
- modules.py
- dataset.py
- train.py
- predict.py
- README.md

### Files Produced From Running Training
Training produces the following files for the MLP:
- gcn_best.pt
- gcn_predictions.csv 
- gcn_accuracy.png 
- gcn_confidence.png
- gcn_confusion_matrix.png
- gcn_loss.png 
- gcn_manual_review.png 
- gcn_reliability.png 
- gcn_tsne.png 

And the following for the GCN:
- mlp_best.pt
- mlp_predictions.csv
- mlp_accuracy.png 
- mlp_confidence.png 
- mlp_confusion_matrix.png
- mlp_loss.png 
- mlp_manual_review.png
- mlp_reliability.png

The code prints test accuracy, macro-F1, multiclass AUROC, per-class precision/recall, 
ECE, inference time, parameter count, GPU memory and failure-case information to the
terminal. When running on the Rangpur cluster these are saved in the resulting .out file.

## Installation Instructions
The project requires Python 3.10 and the libraries listed in the Software 
Dependencies and Versions section.
A Python virtual environment can be created and activated using:
```bash
python3 -m venv ~/gnn_env
source ~/gnn_env/bin/activate
```
The required Python libraries can then be installed using pip:
```bash
python -m pip install torch
python -m pip install torch-geometric
python -m pip install numpy scikit-learn matplotlib
```
For GPU execution, a CUDA-compatible PyTorch installation is required. The appropriate
installation command depends on the CUDA support available on the system.

The Facebook Page-Page Network dataset was obtained from:

https://github.com/gayanku/facebook_page-page_dataset/tree/main

The facebook.npz file must be downloaded and converted into the format expected by 
the PyTorch Geometric dataset loader before running the project.

...

## Reproducability and Random Seeds
A fixed random seed of 42 is used throughout the project to improve reproducibility. 
This ensures that the dataset is divided into the same training, validation and test 
sets each time the code is executed, allowing both the MLP and GCN to be evaluated 
using identical data splits.

The dataset is divided using a stratified 70/15/15 split, preserving approximately the 
same class distribution across the three sets. The same feature standardisation procedure
is applied to both models, with the mean and standard deviation calculated using only 
the training nodes to prevent data leakage.

PyTorch's random seed is also set to 42 before model initialisation and training. 
Both models use the same training settings, including the learning rate, dropout rate 
and maximum number of epochs. The model checkpoint with the lowest validation loss is 
saved and used for final test evaluation.

Although these measures improve reproducibility, small differences in results may 
still occur across different hardware or software environments, particularly when 
using GPU operations.


## Running Training and Inference
### Training the Models
The training script can train both the MLP baseline and the two-layer GCN separately
using the --model argument.

To train the MLP:
```bash
python train.py --model mlp
```
To train the GCN:
```bash
python train.py --model gcn
```

To save the output when running do:
```bash
python train.py --model mlp 2>&1 | tee mlp_training.log
python train.py --model gcn 2>&1 | tee gcn_training.log
```

### Running Inference
The predict.py script is used to perform inference using an already trained MLP or GCN
model. It loads the saved model weights, uses the same dataset preprocessing and node 
splits as training, and predicts the class of the selected Facebook page nodes.

For each node, the script displays the predicted class, prediction confidence, 
probabilities for all four classes, the true class, and whether the prediction is 
correct.

The general command structure is:

```bash
python predict.py [--model {mlp,gcn}] [--node NODE_ID] [--num_samples N] [--threshold VALUE] [--checkpoint PATH]
```
All arguments are optional. The purpose of each argument is:
```
--model:       Selects trained model to use for inference
--node:        Specifies the index of particular node to classify
--num_samples: Specifies how many test nodes to display when --node is not provided
--threshold:   Specifies confidence threshold between 0 and 1 where predictions below
               this threshold are snet for manual review while equal and above are accepted
--checkpoint:  Specifies location of saved model weights if in another file
```
If no arguments are provided, the script uses the trained GCN model and displays 
predictions for the first five nodes in the test set.

Example commands:
```bash
# Using MLP
python predict.py --model mlp

# Classifying a specific node (can be any node in dataset)
python predict.py --model gcn --node 16052

# Classifying 10 test set nodes
python predict.py --model gcn --num_samples 10

# Indicate whether prediction is accepted or sent for manual review using a 0.56
# confidence threshold
python predict.py --model gcn --threshold 0.56

# Using a trained model in a different path
python predict.py --model gcn --checkpoint path/to/gcn_best.pt
```


# Dataset and Preprocessing
## Facebook Page-Page Network Dataset
This project uses the Facebook Large Page-Page Network dataset, which was collected
through the Facebook Graph API in November 2017. The dataset represents a network of
verified Facebook pages, where nodes represent official Facebook pages and edges 
represent mutual likes between pages [2].

The dataset contains four categories of Facebook pages: politicians, governmental
organisations, television shows and companies [2]. These categories were defined by 
Facebook, and the objective of this project is to classify each page into one of 
the four categories using its features and graph connections.

The dataset is available through PyTorch Geometric and was originally provided in 
a NumPy .npz file. The original data required conversion into the format expected 
by the PyTorch Geometric FacebookPagePage loader. This involved extracting the 
node features, edge connections and class labels, and saving them using the 
appropriate array names.

## Node Features, Graph Connections and Class Labels
Each Facebook page is represented by 128-dimensional node features extracted from
descriptions written by the page owners to summarise the purpose of their pages. 
These features provide information about individual Facebook pages and are used 
by both the MLP and GCN during classification.

The graph connections represent mutual likes between Facebook pages. As these 
connections are undirected, the 171,002 edges are stored as 342,004 directed 
edge entries in PyTorch Geometric, representing connections in both directions. 
The dataset does not contain any edge features [3].

Each node has a class label represented by an integer from 0 to 3, corresponding 
to one of the four Facebook page categories. The meaning of each class number is given by:

| Class Number | Facebook Page Category |
|---|---|
| Class 0 | Television Shows |
| Class 1 | Governmental Organisations |
| Class 2 | Companies |
| Class 3 | Politicians |

The feature matrix has dimensions (22470, 128), the graph connections are stored in an 
edge_index tensor with dimensions (2, 342004), and the class labels have dimensions (22470).

The MLP uses only the node features, while the GCN uses both the features and graph 
connections. This allows the GCN to combine information from connected pages, potentially 
improving classification when neighbouring pages belong to similar categories.

## Dataset Statistics and Class Distribution
An initial dataset audit was performed using dataset.py to verify the graph dimensions,
class distribution and validity of the node features and labels. The dataset satistics
are given by:
| Property | Value |
|----------|-------|
| Number of nodes | 22,470 |
| Undirected edges | 171,002 |
| Directed edge entries | 342,004 |
| Features per node | 128 |
| Number of classes | 4 |
| Graph density | 0.001 |
| Graph transitivity | 0.232 |

Source [2] gave values for density and transitivity. The graph is not a Directed 
graph or a Temporal graph and there are no edge features.

The 22,470 nodes are distributed across four classes:
| Class | Number of Nodes | Percentage |
|-------|-----------------|------------|
| Class 0 | 6,495 | 28.91% |
| Class 1 | 6,880 | 30.62% |
| Class 2 | 5,768 | 25.67% |
| Class 3 | 3,327 | 14.81% |
| **Total** | **22,470** | **100.00%** |

From the class distribution is is seen there is some imbalance, with class 1
containing the most amount of nodes and class 3 containing the least. This is
important as a model may achieve high overall accuracy while performing poorly
on less common classes.

Because of this, classification performance will be evaluated using per-class 
precision and recall, macro-F1 and multiclass AUROC in addition to overall accuracy.

The graph density of approximately 0.001 indicates that relatively few connections exist
compared with the total number of possible connections. This indicates that the network 
is sparse, with each page connected to relatively few other pages compared to the total
number of pages. The transitivity of 0.232 indicates that some neighbouring pages form
interconnected groups, which may provide useful information for graph-based classification.

## Training, Validation and Test Splits
The dataset is divided into training, validation and testing sets using a 70/15/15 split.
The splitting procedure is divided by class, meaning that the nodes from each category
are divided separately to preserve approximately the same class distribution across the
three sets.

A 70/15/15 split was chosen instead of the 80/20 training-test split used in the 
original research because this project also requires a separate validation set. 
The validation set is used to select the best model checkpoint based on validation 
loss and determine a confidence threshold for manual review without using the test 
set. This is important because the project investigates confidence reliability, 
and using the test set to make these decisions could produce overly optimistic 
results. The 70/15/15 split provides sufficient data for training while keeping 
separate validation and test sets for model selection and final evaluation.

Using 15% test and 70% training gives 15% leftover for validation while being
really close to the original 80/20 split.

A fixed random seed of 42 is used to ensure that the splits can be reproduced. 
Both the MLP and GCN use the same splits to allow a fair comparison.

The resulting node counts for each split are given by:
| Class | Training | Validation | Test |
|-------|----------|------------|------|
| Class 0 | 4,546 | 974 | 975 |
| Class 1 | 4,816 | 1,032 | 1,032 |
| Class 2 | 4,037 | 865 | 866 |
| Class 3 | 2,328 | 499 | 500 |
| **Total** | **15,727** | **3,370** | **3,373** |

The training set is used to calculate the cross-entropy loss and update the model
parameters. The validation set is used to evaluate performance during training, 
select the checkpoint with the lowest validation loss and establish a confidence
threshold for manual review. The testing set is reserved for final evaluation.

Boolean masks named train_mask, val_mask and test_mask are used to identify the nodes 
belonging to each split. The dataset verification confirmed that no nodes belong 
to more than one split and that every node belongs to exactly one split.

## Feature Standardisation
The node features are standardised to ensure that differences in the scale of individual 
features do not affect the training of the models. Without standardisation, features with 
larger numerical values may have a greater influence on the model than features with 
smaller values, even if they are not more important for classification. Each of the 128 
features is standardised separately by subtracting its mean and dividing by its standard 
deviation, so that the training features have approximately zero mean and unit standard 
deviation.

The mean and standard deviation are calculated using only the training nodes and the same 
transformation is then applied to all nodes in the dataset. This is important because using 
statistics calculated from the validation or test nodes could introduce and leak 
information from these sets into preprocessing and affect the reliability of the 
evaluation. To prevent division by zero, standard deviations smaller than 10^-8 are 
replaced with 10^-8. The same standardised node features are used by both the MLP and 
GCN to ensure a fair and consistent comparison.

## Data Leakage Prevention
This project uses transductive semi-supervised node classification, where the GCN uses the 
features and graph connections of all nodes, including validation and test nodes, but only 
the training node labels are used to update the model parameters. During training, the 
model produces predictions for all nodes in the graph, but the cross-entropy loss is 
calculated only for nodes identified by train_mask. This prevents the validation and test 
labels from directly influencing the model during training while still allowing the GCN to 
learn from the graph structure.

The validation labels are used to monitor model performance, select the checkpoint with the 
lowest validation loss and establish a confidence threshold for manual review. The test 
labels are reserved for final evaluation so that the model can be assessed using labels 
that were not used for training or model selection. This is important because using test 
labels during training or selecting model settings based on test performance could produce 
overly optimistic results that do not accurately represent the model's ability to classify 
unseen labels. The dataset splitting procedure also verifies that the training, validation 
and test masks do not overlap and that every node belongs to exactly one set. Both the MLP 
and GCN use the same splits, ensuring that their performance can be compared fairly.

# Model Implementation and Training
## Multi-Layer Perceptron (MLP) Baseline
The multi-layer perceptron is a neural network model which consists of multiple layers of
neurons. The model is made up of an input layer, hidden layer(s), and an output layer, 
where the neurons in the MLP typically use nonlinear activation functions, allowing the
network to learn complex nonlinear patterns in data. This type of neural network can be 
used to solve both classification and regression problems.

For this project the MLP was chosen as a baseline as it does not consider edge
connections between the nodes, only the node features themselves for classification.
This is useful as it can help evaluate whether graph connections improve classification
and confidence reliability.

A fully connected layer in an MLP takes the input features and transforms them into a new 
set of features using learnable weights and biases. Each neuron in the layer is connected to 
every input feature, where the input values are multiplied by their corresponding weights 
and added together with a bias to produce an output.

An activation function is then applied to the output of the layer to introduce nonlinearity, 
allowing the model to learn more complex patterns in the data. For this project, the ReLU 
activation function is used, which replaces negative values with zero while leaving positive 
values unchanged.

The operation of a fully connected layer followed by an activation function can be 
represented by the following equation:
```math
H^{(l+1)}=\sigma\left(H^{(l)}W^{(l)}+b^{(l)}\right)
```

Where:
* $H^{(l)}$ and $H^{(l+1)}$ represent the input and output node features.
  For the first layer, dimensions are $H^{(l)}$: (22,470 x 128), $H^{(l+1)}$: (22,470 x 32).
  For the second layer, dimensions are $H^{(l)}$: (22,470 x 32), $H^{(l+1)}$: (22,470 x 4).
* $W^{(l)}$ is the learnable weight matrix used to transform the input features.
  For the first layer, the dimensions are (128 x 32).
  For the second layer, the dimensions are (32 x 4).
* $b^{(l)}$ is the learnable bias.
  For the first layer, the dimensions are (32).
  For the second layer, the dimensions are (4).
* $\sigma$ is the activation function, which is ReLU after the first fully connected layer. 
No activation function is applied after the second fully connected layer, as it produces the 
four output logits.

The learnable weights and biases are updated during training so that the MLP can identify 
useful feature combinations for classification. Unlike the GCN, the MLP does not use an 
adjacency or degree matrix because it only considers the features belonging to each 
individual node.

For this project, dropout with a probability of 0.5 is also applied after the ReLU 
activation of the first fully connected layer. This randomly disables some hidden features 
during training to reduce overfitting. The remaining features are then passed to the second 
fully connected layer, which produces four output logits corresponding to the four Facebook 
page categories.

The implemented MLP for this project contains two fully connected layers. The first layer 
receives the 128-dimensional features of each Facebook page as input into its first hidden
layer of 32 neurons and produces 32 hidden features. A ReLU activation function is then 
applied to each neuron output, followed by dropout with a probability of 0.5. The second 
fully connected layer converts the 32 hidden features altered by the activation function 
and dropout, and produces four outputs corresponding to the four Facebook page categories.

The four outputs are logits which represent the model's prediction scores for each class,
where the highest score is the predicted class for the given input. These logits are used 
to calculate the cross-entropy loss during training and are converted into class 
probabilities using softmax during inference and evaluation.

So the MLP has the following architecture:
* One input layer (128 features)
* One hidden layer (32 neurons) with RELU activation and 0.5 dropout
* One ouptut layer (4 neurons)

![Multi-Layer Perceptron Diagram](images/mlp_diagram.jpg)

*Figure 1: Illustration of project MLP architecture.*

## Two-Layer Graph Convolutional Network (GCN)
The Graph Convolutional Network (GCN) is a type of Graph Neural Network (GNN) which uses 
both node features and graph connections to learn information about the nodes. Unlike the 
MLP, which considers each node separately, the GCN uses information from neighbouring nodes 
when calculating its hidden features. This allows the model to learn patterns based on both 
the individual Facebook pages and their connections to other pages. A GCN uses graph
convolutional layers to calculate these features.

A graph convolutional layer combines the features of each node with information from its 
neighbouring nodes to produce a new set of features. The graph convolution operation works 
by first identifying which nodes are connected to each other. These connections are 
represented by an adjacency matrix $A$, where a value of 1 indicates that two nodes are 
connected and a value of 0 indicates that they are not. Self-connections are also added with 
the identity matrix $I$ so that each node includes its own features when calculating the new 
features.

The updated degree matrix $\hat{D}$ is then calculated from $\hat{A}$ by summing the values 
in each row and placing these sums along the diagonal of a new matrix. This means that each 
diagonal value represents the number of connections a node has (i.e. its degree), including 
its self-connection. The updated degree matrix is then used to normalise the contributions 
from neighbouring nodes when their features are combined.

The features from each node and its connected neighbours are then combined using a weighted 
sum. If this is done without degree normalisation then some nodes may have significantly 
more connections than others, which could cause their combined feature values to become 
excessively large. The degree matrix is used to prevent this, where the contributions from 
neighbouring nodes are normalised using the number of connections each node has.

The combined features are then transformed using a learnable weight matrix, which allows the 
model to produce a new set of features. These weights are updated during training so that 
the GCN can learn which combinations of features are useful for classification.

The graph convolution operation can be represented by the following equation:
```math
H^{(l+1)}=\sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)}+b^{(l)}\right)
```

Where:
* $H^{(l)}$ and $H^{(l+1)}$ represent the input and output node features.
  For the first layer dimensions are $H^{(l)}$: (22,470 x 128), $H^{(l+1)}$: (22,470 x 32)
  For the second layer dimensions are $H^{(l)}$: (22,470 x 32), $H^{(l+1)}$: (22,470 x 4)
* $\hat{A}=A+I$ is the adjacency matrix with self-connections added. (22,470 x 22,470)
* $\hat{D}$ is the degree matrix used to normalise the graph connections. (22,470 x 22,470)
* $W^{(l)}$ and $b^{(l)}$ are the learnable weights and bias.
  For the first layer dimensions are $W^{(l)}$: (128 x 32), $b^{(l)}$: (32)
  For the second layer dimensions are $W^{(l)}$: (32 x 4), $b^{(l)}$: (4)
* $\sigma$ is the activation function.

The self-connections allow each node to include its own features, while degree normalisation 
prevents nodes with more connections from having an excessively large influence. The 
learnable weights are updated during training to identify useful feature combinations.

For this project, a two-layer GCN was implemented using the GCNConv layers provided by 
PyTorch Geometric. The first graph convolutional layer receives the 128-dimensional node 
features and produces 32 hidden features for each node. This layer combines the features of 
each node with information from its directly connected neighbours. A ReLU activation 
function and dropout with a probability of 0.5 are then applied before passing the features 
to the second graph convolutional layer.

The second graph convolutional layer receives the 32 hidden features and produces four 
output logits corresponding to the four Facebook page categories. Because the GCN contains 
two graph convolutional layers, information can be passed through up to two graph 
connections, allowing each node's prediction to be influenced by its two-hop neighbourhood.

So the GCN has the structure:
* One input feature representation (128 features per node).
* One hidden graph convolutional layer (32 features per node).
* One output graph convolutional layer (4 features per node).

![Two-Layer GCN Diagram](images/gcn_diagram.jpg)

*Figure 2: Illustration of project GCN architecture.*

## Model Architectures and Hyperparameters Summary
Both the MLP and GCN use the same input feature dimension, hidden feature dimension and 
number of output classes. This allows the two models to be compared using similar 
architectures, with the main difference being that the MLP uses fully connected layers while 
the GCN uses graph convolutional layers.

The architectures of the two models are summarised by the table below:

| Layer | MLP | GCN |
|---|---|---|
| Input | 128 node features | 128 node features and graph connections |
| First Layer | Linear (128 -> 32) | GCNConv (128 -> 32) |
| Activation Function | ReLU | ReLU |
| Regularisation | Dropout (0.5) | Dropout (0.5) |
| Second Layer | Linear (32 -> 4) | GCNConv (32 -> 4) |
| Output | Four class logits | Four class logits |
| Trainable Parameters | 4,260 | 4,260 |

Although both models contain the same number of trainable parameters, the GCN requires 
additional computation because it combines information from neighbouring nodes during each 
graph convolutional layer. The MLP instead processes the features of each node independently.

The hyperparameters used for training both models are:

| Hyperparameter | MLP | GCN |
|---|---|---|
| Input Features | 128 | 128 |
| Hidden Features | 32 | 32 |
| Output Classes | 4 | 4 |
| Number of Trainable Layers | 2 | 2 |
| Activation Function | ReLU | ReLU |
| Dropout Probability | 0.5 | 0.5 |
| Optimiser | Adam | Adam |
| Learning Rate | 0.01 | 0.01 |
| Loss Function | Cross-Entropy Loss | Cross-Entropy Loss |
| Maximum Training Epochs | 1000 | 1000 |
| Random Seed | 42 | 42 |
| Training/Validation/Test Split | 70/15/15 | 70/15/15 |

The same hyperparameters were used for both models to provide a consistent comparison. The 
hidden feature dimension of 32, learning rate of 0.01 and dropout probability of 0.5 were 
selected based on the GCN configuration used by Rozemberczki et al. [1].

The maximum number of training epochs was increased to 1000 to allow both models sufficient 
time to train. However, the model from the final epoch is not necessarily used for 
evaluation, as the checkpoint with the lowest validation loss is selected.

## Training Procedure and Optimisation
The train.py script is used to train both the MLP baseline and two-layer GCN. During 
training, the models learn the weights and biases required to classify Facebook pages by 
comparing their predictions with the correct labels and adjusting their parameters to reduce 
classification error. Both models follow the same training procedure, with the main 
difference being that the GCN also uses graph connections during its forward pass.

There are 7 steps done for training:
* Step 1: Initialise the model and load the dataset
* Step 2: Perform the forward pass
* Step 3: Calculate the cross-entropy loss
* Step 4: Calculate gradients using backpropagation
* Step 5: Update the model parameters using Adam
* Step 6: Evaluate training and validation performance
* Step 7: Repeat training and save the best model

### Model Initialisation and Dataset Loading
The script first sets the random seed to 42 for reproducibility and selects a 
CUDA-compatible GPU if available, otherwise the CPU. The dataset is loaded using 
load_dataset() from dataset.py, which creates the reproducible training, validation and test 
splits and standardises the node features using training-set statistics.

The selected MLP or GCN is then initialised using the hyperparameters above, along with 
the cross-entropy loss function and Adam optimiser with a learning rate of 0.01.

### Forward Pass
During each epoch, the model performs a forward pass to produce predictions for all 22,470 
nodes.

The MLP processes the 128-dimensional node features through two fully connected layers, 
while the GCN also uses the graph connections through two graph convolutional layers. Both 
produce an output matrix of dimensions (22,470 × 4), containing four class logits for each 
node.

Although predictions are generated for all nodes, the training loss is only calculated using
the 15,727 training nodes selected by train_mask.

For the GCN, this follows transductive semi-supervised learning, where the features and 
graph connections of all nodes can be used during training, but validation and test labels 
are excluded from the training loss.

### Cross-Entropy Loss
Cross-entropy loss measures how well the model's predicted probabilities correspond to the 
correct class labels. It is suitable for this project because each Facebook page belongs to 
one of four classes.

The four output logits can be converted into class probabilities using the softmax function:
```math
p_{i,c}=\frac{e^{z_{i,c}}}{\sum_{j=0}^{3}e^{z_{i,j}}}
```
Where:
* $z_{i,c}$ is the output logit for class $c$ of node $i$.
* $p_{i,c}$ is the predicted probability of node $i$ belonging to class $c$.
* The denominator ensures that the four class probabilities sum to 1.

The average cross-entropy loss is then calculated using:
```math
L=-\frac{1}{N}\sum_{i=1}^{N}\log(p_{i,y_i})
```
Where:
* $L$ is the average cross-entropy loss.
* $N$ is the number of training nodes (15,727).
* $y_i$ is the correct class label for node $i$.
* $p_{i,y_i}$ is the probability assigned to the correct class.

The loss is smaller when the model assigns a high probability to the correct class and 
larger when it assigns a low probability.

The script uses PyTorch's nn.CrossEntropyLoss(), which calculates the loss directly from the 
logits using an internal log-softmax operation. Therefore, softmax does not need to be 
applied separately during training.

### Backpropagation
Once the loss is calculated, backpropagation determines how each learnable weight and bias 
contributes to the loss by calculating its gradient:
```math
\frac{\partial L}{\partial\theta}
```
Where:
* $L$ is the cross-entropy loss.
* $\theta$ represents a learnable weight or bias.

The gradient describes how the loss changes as the parameter changes.

The script uses loss.backward() to calculate these gradients through the model using the 
chain rule. The gradients indicate how the parameters should be adjusted to reduce the loss.

Before each backward pass, optimizer.zero_grad() clears the gradients from the previous 
epoch, preventing unwanted gradient accumulation.

### Parameter Optimisation Using Adam
The Adam optimiser uses the calculated gradients to update the model's weights and biases, 
with the aim of reducing the cross-entropy loss over successive epochs.

The basic principle of gradient descent can be represented by:
```math
\theta_{t+1}=\theta_t-\eta\frac{\partial L}{\partial\theta_t}
```
Where:
* $\theta_t$ and $\theta_{t+1}$ are the parameter values before and after the update.
* $\eta$ is the learning rate (0.01).
* $\frac{\partial L}{\partial\theta_t}$ is the gradient of the loss with respect to the parameter.

The learning rate controls the size of the parameter updates. A large learning rate may 
cause unstable training, while a very small learning rate may slow convergence.

For this project, Adam is used instead of basic gradient descent. Adam maintains moving 
averages of gradients and squared gradients to adapt parameter updates, which can improve 
training efficiency and stability.

The updates are performed using optimizer.step() after backpropagation.

### Training and Validation Evaluation
After each parameter update, the model is evaluated on the training and validation sets 
using the evaluate() function.

The script uses model.eval() to disable dropout and torch.no_grad() to prevent gradient 
calculations. Cross-entropy loss and classification accuracy are then calculated separately 
for each set.

The predicted class is determined by selecting the highest output logit, and accuracy is 
calculated using:
```math
\text{Accuracy}=\frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
```

The validation set is used to measure performance on nodes whose labels were not used for 
parameter updates, helping identify whether the model generalises beyond the training data 
or begins to overfit.

Training and validation losses and accuracies are recorded after every epoch and used to 
generate the corresponding performance graphs.


### Training Epochs and Model Checkpointing
Steps 2–6 are repeated for 1000 epochs, with each epoch performing one parameter update 
using the entire set of 15,727 training nodes.

After each epoch, the validation loss is compared with the lowest previously recorded value. 
If it improves, the model's parameters are saved to mlp_best.pt or gcn_best.pt.

Training continues for all 1000 epochs without early stopping. However, the checkpoint with 
the lowest validation loss is selected rather than automatically using the parameters from 
the final epoch.

After training, the best checkpoint is loaded and used for the final test-set evaluation, 
confidence analysis and other performance measurements.

## Validation and Best Model Selection
| Metric | MLP | GCN |
|---|---:|---:|
| Maximum Training Epochs | 1000 | 1000 |
| Best Epoch | 182 | 297 |
| Training Loss | 0.4500 | 0.1395 |
| Training Accuracy | 83.84% | 95.86% |
| Validation Loss | 0.5792 | 0.2328 |
| Validation Accuracy | 79.70% | 93.65% |

Examining the validation results, it is observed that the MLP achieved its lowest 
validation loss of 0.5792 at epoch 182, while the GCN achieved a significantly lower 
validation loss of 0.2328 at epoch 297. This lower loss over a higher epoch count indicates
that the GCN has more potential to learn more complex patterns. The GCN also achieved a 
higher validation accuracy of 93.65% compared to 79.70% for the MLP, indicating that the 
GCN is more effective at classifying the Facebook pages.

This difference in performance suggests that using graph connections allows the GCN 
to learn more useful information than the MLP, which only considers the individual node 
features. Thus, the models saved at epochs 182 and 297 were selected for the final 
evaluation, as these achieved the lowest validation losses during training.

## Training and Validation Loss Curves
![MLP Loss Diagram](images/mlp_loss.jpg)

*Figure 3: MLP training and validation cross-entropy loss across 1000 epochs.*

![GCN Loss Diagram](images/gcn_loss.jpg)

*Figure 4: GCN training and validation cross-entropy loss across 1000 epochs.*

Examining Figures 3 and 4, it is observed that both the MLP and GCN initially experience a 
significant decrease in training and validation loss, indicating that both models are 
learning useful patterns during the early stages of training.

For the MLP in Figure 3, the training loss continues to decrease throughout the 1000 
epochs, while the validation loss begins to increase after approximately 180 epochs. This 
suggests that the MLP begins to overfit the training data, as it continues improving its 
performance on the training nodes while becoming less effective on the validation nodes.
The training loss reducing while validation increaseing widens the generalisation gap, 
meaning the model becomes less generalised and overfits more to the training data as 
training continues.

For the GCN in Figure 4, both losses decrease significantly before the validation loss 
begins to stabilise at approximately 0.23. Although the validation loss increases slightly 
towards the later epochs, the increase is significantly smaller than that of the MLP, 
suggesting that the GCN experiences less severe overfitting.

Overall, the GCN achieves a much lower training and validation loss than the MLP, 
indicating that incorporating the graph connections improves the model's ability to learn 
useful patterns while also providing better generalisation to the validation nodes. This 
also supports selecting the best model checkpoint based on validation loss rather than 
using the final epoch.

## Training and Validation Accuracy Curves
![MLP Accuracy Diagram](images/mlp_accuracy.jpg)

*Figure 5: MLP training and validation accuracy across 1000 epochs.*

![GCN Accuracy Diagram](images/gcn_accuracy.jpg)

*Figure 6: GCN training and validation accuracy across 1000 epochs.*

Examining Figures 5 and 6, it is observed that both the MLP and GCN experience a 
significant increase in training and validation accuracy during the early epochs, before 
the rate of improvement begins to decrease.

For the MLP in Figure 5, the training accuracy continues to increase throughout the 1000 
epochs, reaching approximately 87%, while the validation accuracy begins to stabilise at 
approximately 82%. The increasing difference between the training and validation accuracy 
suggests that the MLP experiences some overfitting, as it continues to improve on the 
training nodes without achieving similar improvements on the validation nodes.

For the GCN in Figure 6, both training and validation accuracy increase significantly 
before stabilising at approximately 97% and 94%, respectively. The difference between the 
training and validation accuracy remains relatively small compared to the MLP, indicating 
that the GCN generalises better to the validation nodes.

Overall, the GCN achieves significantly higher training and validation accuracy than the 
MLP, suggesting that incorporating the graph connections allows the model to learn more 
useful patterns for classifying Facebook pages. Although both models show a difference 
between training and validation accuracy, the GCN maintains higher validation performance 
with a smaller gap between the two curves.

# Example Inputs, Outputs and Visualisations
## Example Input Graph and Node Features
To demonstrate the input data used by the models, node 0 was selected from the test set. 
The first 10 standardised feature values of this node are shown below, along with its 
connected neighbouring node.

| Property | Value |
|---|---|
| Node ID | 0 |
| Actual Class | 3 (Politicians) |
| First 10 Standardised Features | [-0.8766, -0.0561, -0.0574, -0.0610, 0.1523, 0.1374, 0.3039, 0.7328, 1.9887, -1.2992] |
| Number of Neighbours | 1 |
| Neighbouring Node ID | 18427 |

Examining the input, it is observed that node 0 contains both positive and negative 
standardised feature values, with only one neighbouring node connected to it. The MLP uses 
the node features for classification, while the GCN can also use information from node 
18427 when calculating its hidden features.

## Example Node Classification Predictions
To demonstrate the classification performance of the trained models, the predict.py script 
was used to generate predictions for five nodes from the test set. The predicted classes 
and confidence values of the MLP and GCN are compared below.

| Node ID | Actual Class | MLP Prediction | MLP Confidence | GCN Prediction | GCN Confidence |
|---|---|---|---|---|---|
| 0 | 3 | 0 (Incorrect) | 37.28% | 3 (Correct) | 97.74% |
| 7 | 2 | 2 (Correct) | 89.61% | 2 (Correct) | 98.71% |
| 10 | 1 | 1 (Correct) | 96.26% | 1 (Correct) | 79.64% |
| 18 | 1 | 1 (Correct) | 96.01% | 1 (Correct) | 99.84% |
| 35 | 0 | 0 (Correct) | 44.20% | 0 (Correct) | 99.58% |

Examining these results, it is observed that the GCN correctly classifies all five selected 
nodes, while the MLP incorrectly classifies node 0 as class 0 instead of class 3. The GCN 
also generally produces higher prediction confidence, particularly for nodes 0 and 35.

This suggests that incorporating information from neighbouring nodes can improve 
classification compared to using only the individual node features. However, these five 
examples alone are not sufficient to determine the overall effectiveness of the models, 
which is evaluated using the complete test set.

## Class Probabilities and Prediction Confidence
The class probabilities of node 0 were examined to compare how the MLP and GCN classify the
same Facebook page. The probabilities produced by both models are shown below.

| Class | MLP Probability | GCN Probability |
|---|---:|---:|
| Class 0 (Television Shows) | 37.28% | 1.97% |
| Class 1 (Governmental Organisations) | 19.73% | 0.00% |
| Class 2 (Companies) | 8.99% | 0.29% |
| Class 3 (Politicians) | 34.00% | 97.74% |
| **Predicted Class** | **0 (Incorrect)** | **3 (Correct)** |
| **Prediction Confidence** | **37.28%** | **97.74%** |

Examining the probabilities, it is observed that the MLP assigns similar probabilities to 
classes 0 and 3, with class 0 receiving the highest probability of 37.28%. This results in 
an incorrect prediction with relatively low confidence.

The GCN however assigns 97.74% probability to class 3, which is the correct category of 
node 0. This indicates that the GCN is significantly more confident in its prediction for 
this particular node.

However, a high prediction confidence does not necessarily mean that the classification is 
correct. To demonstrate this, node 16052 was also examined, where both models incorrectly 
classify the node as class 2 instead of class 1.

| Class | MLP Probability | GCN Probability |
|---|---:|---:|
| Class 0 (Television Shows) | 1.54% | 0.00% |
| Class 1 (Governmental Organisations) | 5.56% | 0.00% |
| Class 2 (Companies) | 91.20% | 100.00% |
| Class 3 (Politicians) | 1.69% | 0.00% |
| **Predicted Class** | **2 (Incorrect)** | **2 (Incorrect)** |
| **Prediction Confidence** | **91.20%** | **100.00%** |

Examining these results, it is observed that both models incorrectly classify node 16052 
despite having high prediction confidence. The GCN is particularly overconfident, assigning 
approximately 100% probability to the incorrect class.

This demonstrates that although the GCN can achieve higher classification accuracy, it can 
still produce highly confident incorrect predictions. This is one of the main issues 
investigated in the confidence reliability.

## Learned Node Embeddings t-SNE
![GCN t-SNE Diagram](images/gcn_tsne.jpg)

*Figure 7: Two-dimensional t-SNE visualisation of learned GCN node embeddings, coloured by ground-truth class.*

The learned node embeddings of the GCN were visualised using t-distributed Stochastic 
Neighbour Embedding (t-SNE). This technique reduces high-dimensional features into two 
dimensions, allowing the relationships between nodes to be visualised. For this project, 
4,000 nodes were randomly selected for t-SNE visualisation and their 32-dimensional hidden 
features from the first GCN layer were reduced to two dimensions using t-SNE.

Examining Figure 7, it is observed that the nodes form several clusters based on their 
ground-truth classes. Class 0 (Television Shows) is mostly concentrated towards the right, 
Class 1 (Governmental Organisations) towards the lower left, Class 2 (Companies) towards 
the upper left, and Class 3 (Politicians) towards the upper right.

Although the classes form relatively distinct groups, there is still some overlap between 
them, particularly around the centre of the graph. This suggests that the GCN has learned 
useful feature representations which separate the four Facebook page categories, although 
some nodes remain difficult to distinguish based on their learned features.

## Confusion Matrices
![MLP Confusion Matrix](images/mlp_confusion_matrix.jpg)

*Figure 8: Confusion matrix showing the MLP's predicted and actual classes on the test set.*

![GCN Confusion Matrix](images/gcn_confusion_matrix.jpg)

*Figure 9: Confusion matrix showing the GCN's predicted and actual classes on the test set.*

Examining Figures 8 and 9, it is observed that both the MLP and GCN correctly classify the 
majority of test nodes, as shown by the larger values along the diagonal of the confusion 
matrices. However, the GCN achieves significantly more correct classifications across all 
four classes compared to the MLP.

For the MLP in Figure 8, Class 3 (Politicians) has the poorest classification performance, 
with only 309 out of 500 nodes correctly classified. A significant number of these nodes 
are incorrectly classified as Class 0 (Television Shows), with 93 incorrect predictions. 
Another noticeable error occurs for Class 2 (Companies), where 129 nodes are incorrectly 
classified as Class 1 (Governmental Organisations).

For the GCN in Figure 9, the number of correct classifications increases across every 
class. In particular, Class 3 improves from 309 to 441 correctly classified nodes, while 
the incorrect classifications of Class 2 as Class 1 decrease from 129 to 39. This indicates 
that the GCN is more effective at distinguishing between the Facebook page categories.

Overall, the GCN produces fewer classification errors than the MLP, suggesting that 
incorporating graph connections provides useful additional information for classification. 
However, some confusion between the classes remains, which will be examined further in the 
next sections.

# Investigation of the Open Research Dilemma
## Research Question: Confidence and Ambiguity

## Quantitative Classification Benchmarking
### Experimental Setup and Fair Comparison

### Accuracy and Macro-F1 Comparison
| Metric | MLP | GCN |
|---|---:|---:|
| Test Accuracy | 80.55% | 93.30% |
| Macro-F1 | 0.7875 | 0.9292 |

### Per-Class Precision and Recall
| Class | MLP Precision | MLP Recall | MLP F1 | GCN Precision | GCN Recall | GCN F1 |
|---|---:|---:|---:|---:|---:|---:|
| Class 0 | 0.8443 | 0.8564 | 0.8503 | 0.9356 | 0.9385 | 0.9370 |
| Class 1 | 0.7914 | 0.8566 | 0.8227 | 0.9293 | 0.9428 | 0.9360 |
| Class 2 | 0.8106 | 0.7956 | 0.8030 | 0.9413 | 0.9446 | 0.9429 |
| Class 3 | 0.7410 | 0.6180 | 0.6739 | 0.9207 | 0.8820 | 0.9009 |


### Multiclass AUROC Comparison
| Metric | MLP | GCN |
|---|---:|---:|
| Macro AUROC (One-vs-Rest) | 0.9482 | 0.9880 |

### Classification Performance Discussion

## Confidence Reliability and Calibration
### Confidence on Correct and Incorrect Predictions
| Metric | MLP | GCN |
|---|---:|---:|
| Mean Confidence (Correct Predictions) | 83.40% | 94.84% |
| Mean Confidence (Incorrect Predictions) | 57.41% | 74.14% |

![MLP Confidence Distribution](images/mlp_confidence.jpg)

*Figure 10: Distribution of MLP prediction confidence for correctly and incorrectly classified test nodes.*

![GCN Confidence Distribution](images/gcn_confidence.jpg)

*Figure 11: Distribution of GCN prediction confidence for correctly and incorrectly classified test nodes.*

### Expected Calibration Error (ECE)
| Metric | MLP | GCN |
|---|---:|---:|
| Expected Calibration Error (ECE) | 0.0427 | 0.0146 |


### Reliability Diagrams and Calibration Analysis
![MLP Reliability Diagram](images/mlp_reliability.jpg)

*Figure 12: MLP reliability diagram comparing average prediction confidence with observed classification accuracy.*

![GCN Reliability Diagram](images/gcn_reliability.jpg)

*Figure 13: GCN reliability diagram comparing average prediction confidence with observed classification accuracy.*

### Overconfidence and Ambiguous Predictions

## Confidence Threshold and Manual Review
### Confidence Threshold Selection
There needs to be a balance between workout and prediction accuracy.

for GCN target accuracy: 0.95
for 

| Metric | MLP | GCN |
|---|---:|---:|
| Target Accepted Accuracy (Validation) | 95% | 95% |
| Selected Confidence Threshold | 0.86 | 0.56 |
| Validation Accepted Accuracy | 95.36% | 95.02% |


### Automatically Accepted Prediction Accuracy
| Metric | MLP | GCN |
|---|---:|---:|
| Overall Test Accuracy | 80.55% | 93.30% |
| Accepted Test Prediction Accuracy | 95.73% | 94.51% |


### Manual Review Workload
| Metric | MLP | GCN |
|---|---:|---:|
| Automatically Accepted Predictions | 48.65% | 97.15% |
| Predictions Sent for Manual Review | 51.35% | 2.85% |


### Accuracy–Workload Trade-off
![MLP Manual Review Trade-off](images/mlp_manual_review.jpg)

*Figure 14: MLP trade-off between automatically accepted prediction accuracy and the percentage of test nodes sent for manual review.*

![GCN Manual Review Trade-off](images/gcn_manual_review.jpg)

*Figure 15: GCN trade-off between automatically accepted prediction accuracy and the percentage of test nodes sent for manual review.*

## Computational Resource Profiling
### Model Parameter Count

### Inference Runtime and Latency

### Peak GPU Memory Usage

### Computational Efficiency Comparison
| Metric | MLP | GCN |
|---|---:|---:|
| Model Parameters | 4,260 | 4,260 |
| Average Inference Time (CPU) | 1.499 ms | 36.090 ms |
| Peak GPU Memory | Not measured | Not measured |


## Qualitative Failure Analysis
### Failure Case Selection Method
| Model | Node ID | Actual Class | Predicted Class | Confidence |
|---|---:|---:|---:|---:|
| MLP | 2924 | 0 | 1 | 99.99% |
| MLP | 12241 | 3 | 2 | 99.99% |
| MLP | 5047 | 3 | 0 | 99.87% |
| MLP | 17334 | 0 | 3 | 99.82% |
| MLP | 10198 | 1 | 2 | 99.80% |
| GCN | 16052 | 1 | 2 | 100.00% |
| GCN | 9254 | 0 | 3 | 100.00% |
| GCN | 9708 | 3 | 0 | 100.00% |
| GCN | 20212 | 0 | 3 | 100.00% |
| GCN | 6163 | 1 | 2 | 100.00% |


### Failure Case 1

### Failure Case 2

### Failure Case 3

### Failure Case 4

### Failure Case 5

### Common Failure Modes and Root Causes

## Engineering Recommendation and Trade-offs
### Classification Accuracy versus Confidence Reliability

### Performance versus Computational Cost

### MLP versus GCN: Recommended Model

### Deployment Recommendations and Limitations

# Conclusion
## Summary of Findings

## Acceptance Criteria Evaluation

## Limitations and Future Improvements

# Artificial Intelligence Usage Disclosure
## AI Tools Used

## How AI Was Used

## Verification and Independent Work


# Feasability Review (Checked off on 6/10/2026)
## User Need, Scope & Acceptance Criteria
This multi-layer GNN project will develop a two-layer Graph Convolutional Network (GCN) 
to classify Facebook pages as politicians, governmental organisations, television shows 
or companies. The intended users are social network analysts seeking to reduce manual classification. 
The project will investigate whether the GCN produces more reliable confidence scores on ambiguous 
predictions than a Multi-Layer Perceptron (MLP) baseline. The acceptance criteria are:
1.	Classification: Achieve reasonable accuracy and compare macro-F1, per-class precision/recall and multiclass AUROC against the MLP.
2.	Confidence: Compare confidence on correct and incorrect predictions and measure Expected Calibration Error (ECE).
3.	Manual review: Establish a confidence threshold and measure accepted-prediction accuracy and manual review workload.
4.	Resources: Compare inference time, parameter count and peak GPU memory, targeting under 2 GB VRAM and 5 seconds for test-set inference.
5.	Failure analysis: Investigate at least three ambiguous or incorrectly classified nodes.

## Model Choice & Course Concepts
The Facebook Page-Page Network contains 22,470 nodes, 171,002 undirected connections, 
128-dimensional features and four classes. A two-layer GCN implemented in PyTorch 
Geometric will combine node features with information from up to two-hop neighbours. 
An MLP using only node features will serve as the baseline to evaluate whether graph 
connections improve classification and confidence reliability. Rozemberczki et al. [1] 
reported a GCN micro-F1 of 0.932 on this dataset. Their configuration of 32 hidden filters, 
Adam, learning rate 0.01 and dropout 0.5 will be used initially. The project is classified 
as Normal Difficulty and applies graph convolutions, feature representations, 
regularisation and model evaluation. Transductive semi-supervised learning will be used, 
with test labels excluded from training. Learned embeddings will be visualised using t-SNE 
or UMAP.

## Preliminary Feasibility Evidence
An initial dataset audit was completed using PyTorch Geometric. The dataset loaded 
successfully with 22,470 nodes, 342,004 directed edge entries, 128 features per node 
and four classes. The class distribution ranged from 3,327 nodes (14.81%) to 6,880 
nodes (30.62%), indicating some class imbalance. The audit confirms that the dataset 
was successfully loaded and contains the expected node features and classification 
labels. The uneven class distribution also shows why it is important to evaluate 
performance across individual classes rather than relying on overall accuracy. Model 
training and confidence reliability have not yet been tested.

## Risks, Budget & Fallback
The main risks are misleading graph connections, overconfident predictions and label 
leakage. These will be addressed through held-out evaluation, confidence analysis and 
training-label masking. The initial GCN will train for up to 200 epochs within a 
proposed 30-minute training budget and 2 GB VRAM limit. Runtime and memory usage 
will be measured. The next experiment will establish reproducible train/validation/test 
splits and train an MLP baseline. If GCN performance is poor, hyperparameters will be 
adjusted. If it offers little improvement, the MLP will remain the simpler fallback.

# References:
[1] B. Rozemberczki, C. Allen, and R. Sarkar, “Multi-Scale Attributed Node Embedding,” Journal of Complex Networks, 2021. [Online]. Available: https://arxiv.org/abs/1909.13021

[2] Stanford Network Analysis Project (SNAP), “Facebook Large Page-Page Network,” Stanford University. [Online]. Available: https://snap.stanford.edu/data/facebook-large-page-page-network.html

[3] PyTorch Geometric, “torch_geometric.datasets.FacebookPagePage,” PyTorch Geometric Documentation, Version 2.6.0. [Online]. Available: https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.FacebookPagePage.html
















