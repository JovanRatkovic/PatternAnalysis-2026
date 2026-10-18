# COMP3710 Project: Multi-layer Graph Neural Network for Classifying Facebook Pages

## The Open Research Question
In web-scale graph classification, models frequently achieve deceptively high aggregate
test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, 
or out-of-distribution cases.

This project will



## The Proposed Solution

## How the Algorithm Works

# Feasability Review
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

## References:
[1] B. Rozemberczki, C. Allen, and R. Sarkar, “Multi-Scale Attributed Node Embedding,” 
Journal of Complex Networks, 2021. [Online]. Available: https://arxiv.org/abs/1909.13021

# Dependencies and Reproducability
## Software Dependencies and VErsions

## Hardware and Computing Environment

## Project File Structure

## Installation Instructions

## Reproducability and Random Seeds

## Running Training and Inference

# Dataset and Preprocessing

## Facebook Page-Page Network Dataset

## Node Features, Graph Connections and Class Labels

## Dataset Statistics and Class Distribution

## Training, Validation and Test Splits

## Feature Standardisation

## Data Leakage Prevention

# Model Implementation and Training
## Multi-Layer Perceptron (MLP) Baseline
The multi-layer perceptron is a neural network model which consists of multiple layers of
neurons. The model is made up of an input layer, hidden layer(s), and an output layer, 
and can be used to solve both classification and regression problems.


## Two-Layer Graph Convolutional Network (GCN)

## Model Architectures and Hyperparameters

## Training Procedure and Optimisation

## Validation and Best Model Selection

## Training and Validation Loss Curves

## Training and Validation Accuracy Curves

# Example Inputs, Outputs and Visualisations
## Example Input Graph and Node Features

## Example Node Classification Predictions

## Class Probabilities and Prediction Confidence

## Learned Node Embeddings (t-SNE or UMAP)

## Confusion Matrices

# Investigation of the Open Research Dilemma
## Research Question: Confidence and Ambiguity

## Quantitative Classification Benchmarking
### Experimental Setup and Fair Comparison

### Accuracy and Macro-F1 Comparison

### Per-Class Precision and Recall

### Multiclass AUROC Comparison

### Classification Performance Discussion

## Confidence Reliability and Calibration
### Confidence on Correct and Incorrect Predictions

### Expected Calibration Error (ECE)

### Reliability Diagrams and Calibration Analysis

### Overconfidence and Ambiguous Predictions

## Confidence Threshold and Manual Review
### Confidence Threshold Selection

### Automatically Accepted Prediction Accuracy

### Manual Review Workload

### Accuracy–Workload Trade-off

## Computational Resource Profiling
### Model Parameter Count

### Inference Runtime and Latency

### Peak GPU Memory Usage

### Computational Efficiency Comparison

## Qualitative Failure Analysis
### Failure Case Selection Method

### Failure Case 1

### Failure Case 2

### Failure Case 3

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

# References






















