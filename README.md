# Hierarchical_FFC
Title
Hierarchical Deep Learning Framework for File Fragment Classification

Description
This project implements a hierarchical deep learning framework for classifying file fragments. The framework first constructs a tree structure through hierarchical clustering of category mean features, then trains specialized classifiers for each leaf node in the tree. This approach enhances classification accuracy by leveraging hierarchical relationships between file types and optimizing model complexity for different categories. The system supports multiple classifier architectures (CNN, LSTM, CNN-LSTM) and includes tools for result visualization and performance evaluation.

Dataset Information
Dataset Name: File Fragment Type (FFT) - 75 Dataset
Source: IEEE Dataport, accessible via DOI: http://dx.doi.org/10.21227/kfxw-8084
Dataset Authors: Govind Mittal (New York University), Pawel Korus (New York University), Nasir Memon (New York University)
Publication Date: 2022-05-18

Code Information
Core Components
ClassificationReportCallback: Custom Keras callback to track and report classification performance during training
hierarchical_clustering: Builds a 3-level clustering tree using category mean features
train_classifiers: Trains specialized classifiers for each leaf node in the clustering tree

Classifier Architectures:
P1: Deep CNN with 4 convolutional layers and Global Average Pooling
P2: Lightweight CNN with 2 convolutional layers
Byte-level LSTM: Recurrent model using LSTM layers
CNN-LSTM: Hybrid model combining CNN and GRU layers
Tree Visualization: Generates a graphical representation of the hierarchical clustering structure

Key Functions
map_labels: Maps original labels from the FFT-75 dataset to local indices for leaf node classifiers
merge_leaf_into_sibling_leaf: Merges small leaf nodes (<4 categories) to ensure sufficient training data
split_leaf_node: Splits large leaf nodes (>15 categories) to maintain classification precision
visualize_tree: Creates a PNG visualization of the hierarchical clustering tree

Usage Instructions

1. Environment Setup
Install required dependencies (see Requirements section)

2. Dataset Preparation
Download the FFT-75 dataset from IEEE Dataport via the link: http://dx.doi.org/10.21227/kfxw-8084
Select the 512-byte block variant (consistent with the project’s input feature length)
Extract the training, validation, and testing subsets (following the dataset’s 80-10-10 split)
Organize the data into NPZ files (features: x, labels: y) and place them in the ./512_1/ directory (or adjust the file path in the code to match your local storage)

3. Run the Framework
Execute the main script:
python [script_name].py

4. Output Files
Tree Visualization: tree_structure.png (hierarchical clustering tree)
Model Architectures: model_P1.png and model_P2.png (classifier architectures)
Training Logs: Classification reports and accuracy metrics printed during training

5.  Others
If you need to test 4096-byte file fragments, simply replace the dataset with the 4096-byte version.

Requirements

Python 3.6+
NumPy
scikit-learn
TensorFlow 2.x
Keras
Graphviz
pydot (for model visualization)
Methodology

1. Data Preprocessing
Load the FFT-75 dataset (512-byte block variant) training and validation subsets
Compute mean feature vectors for each of the 75 file type categories

2. Hierarchical Clustering
Level 1: Cluster all 75 category means into 2 initial clusters
Level 2: Cluster each Level 1 cluster into [2, 3] sub-clusters
Level 3: Cluster each Level 2 cluster into specified sub-clusters (configurable)
Node Adjustment: Merge small nodes (<4 categories) and split large nodes (>15 categories) to optimize classifier performance

3. Classifier Training
For each leaf node in the clustering tree:
Filter training/validation data for the node's categories (from the FFT-75 dataset)
Map original dataset labels to local indices
Train specialized classifier (P1, P2, LSTM, or CNN-LSTM)
Evaluate performance using custom callback

4. Result Visualization
Generate hierarchical tree visualization
Print classification reports for best-performing epochs

License & Contribution Guidelines
License: The FFT-75 dataset follows IEEE Dataport’s usage terms. This project is for research purposes only; commercial use requires compliance with the dataset’s license and prior permission.
