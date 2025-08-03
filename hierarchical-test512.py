import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Conv1D, LeakyReLU, MaxPool1D, GlobalAveragePooling1D, Dropout, Dense, LSTM, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from graphviz import Digraph
import sys 
from tensorflow.python.keras.utils.vis_utils import plot_model
sys.setrecursionlimit(300)

class ClassificationReportCallback(Callback):
    def __init__(self, x_val, y_val, num_classes):
        self.x_val = x_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.best_epoch = -1
        self.best_accuracy = 0
        self.best_report = None
        self.best_correct_counts = None

    def on_epoch_end(self, epoch, logs=None):
        y_val_cat = np.argmax(self.y_val, axis=1)
        y_pred = self.model.predict(self.x_val)
        y_pred_cat = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_val_cat, y_pred_cat)   
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch + 1
            self.best_report = classification_report(y_val_cat, y_pred_cat, target_names=[f"Class {i}" for i in range(self.num_classes)])
            
            correct_counts = {f"Class {i}": 0 for i in range(self.num_classes)}
            for true, pred in zip(y_val_cat, y_pred_cat):
                if true == pred:
                    correct_counts[f"Class {true}"] += 1
            self.best_correct_counts = correct_counts
                     
    def on_train_end(self, logs=None):
        if self.best_report is not None:
            print(f"\nBest Classification Report for Epoch {self.best_epoch}:")
            print(self.best_report)
            print("Correctly Classified Counts:")
            for class_name, count in self.best_correct_counts.items():
                print(f"{class_name}: {count}")

train_data = np.load('./512_1/train.npz')
x_train = train_data['x']
y_train = train_data['y']

val_data = np.load('./512_1/val.npz')
x_val = val_data['x']
y_val = val_data['y']

unique_labels = np.unique(y_train)
category_means = np.array([np.mean(x_train[y_train == label], axis=0) for label in unique_labels])

def print_category_means_for_labels(category_means, unique_labels, y_train, target_labels):
    for label in target_labels:
        if label in unique_labels:
            mean_vector = category_means[label]
            print(f"标签: {label} 的均值特征向量:\n{mean_vector}\n")
        else:
            print(f"标签: {label} 不存在于数据集中。")

target_labels = [14, 13]
print_category_means_for_labels(category_means, unique_labels, y_train, target_labels)

n_clusters_level_1 = 2  
n_clusters_level_2 = [2, 3]  
n_clusters_level_3 = [[2, 1], [4, 1, 2]]  

tree_structure = defaultdict(lambda: defaultdict(dict))

def hierarchical_clustering(category_means, n_clusters_level_1, n_clusters_level_2, n_clusters_level_3):
    print("Starting Level 1 Clustering...")
    agglom_level_1 = AgglomerativeClustering(n_clusters=n_clusters_level_1)
    level_1_clusters = agglom_level_1.fit_predict(category_means)
    print(f"Level 1 Clustering Result: {level_1_clusters}\n")

    for i in range(n_clusters_level_1):  
        indices = np.where(level_1_clusters == i)[0]
        cluster_means = category_means[indices]
        
        tree_structure[f'Level_1_{i}'] = {}

        print(f"Starting Level 2 Clustering for Level 1 Cluster {i} with {n_clusters_level_2[i]} clusters...")
        agglom_level_2 = AgglomerativeClustering(n_clusters=n_clusters_level_2[i])
        level_2_cluster_result = agglom_level_2.fit_predict(cluster_means)
        print(f"Level 2 Clustering Result for Level 1 Cluster {i}: {level_2_cluster_result}\n")

        for j in range(n_clusters_level_2[i]):
            sub_indices = indices[np.where(level_2_cluster_result == j)[0]]
            cluster_means_level_2 = category_means[sub_indices]
            leaf_labels = [unique_labels[label] for label in sub_indices]

            if len(leaf_labels) == 1:
                print(f"Level_2_{i * 10 + j} contains only one category, attempting to merge.")
                tree_structure[f'Level_1_{i}'][f'Level_2_{i * 10 + j}'] = {'Single_Leaf': leaf_labels}
                merge_leaf_into_sibling_leaf(tree_structure, f'Level_1_{i}', f'Level_2_{i * 10 + j}', leaf_labels)
                continue
            
            print(f"Starting Level 3 Clustering for Level 2 Cluster {i * 10 + j} with {n_clusters_level_3[i][j]} clusters...")
            agglom_level_3 = AgglomerativeClustering(n_clusters=n_clusters_level_3[i][j])
            level_3_cluster_result = agglom_level_3.fit_predict(cluster_means_level_2)
            print(f"Level 3 Clustering Result for Level 2 Cluster {i * 10 + j}: {level_3_cluster_result}\n")

            tree_structure[f'Level_1_{i}'][f'Level_2_{i * 10 + j}'] = {}

            for k in range(n_clusters_level_3[i][j]):
                final_indices = sub_indices[np.where(level_3_cluster_result == k)[0]]
                leaf_labels = [unique_labels[label] for label in final_indices]
                
                if len(leaf_labels) < 4:
                    print(f"Leaf node Level_3_{i * 100 + j * 10 + k} has fewer than 4 categories, attempting to merge.")
                    merge_leaf_into_sibling_leaf(tree_structure, f'Level_1_{i}', f'Level_2_{i * 10 + j}', leaf_labels)
                elif len(leaf_labels) > 15:
                    print(f"Leaf node Level_3_{i * 100 + j * 10 + k} has more than 15 categories, attempting to split.")
                    split_leaf_node(tree_structure, f'Level_1_{i}', f'Level_2_{i * 10 + j}', f'Level_3_{i * 100 + j * 10 + k}', leaf_labels)
                else:
                    tree_structure[f'Level_1_{i}'][f'Level_2_{i * 10 + j}'][f'Level_3_{i * 100 + j * 10 + k}'] = leaf_labels

    return tree_structure

def map_labels(y, leaf_labels):
    mapped_y = np.zeros_like(y)
    for idx, labels in enumerate(leaf_labels):
        mask = np.isin(y, labels)
        mapped_y[mask] = idx
    return mapped_y

def merge_leaf_into_sibling_leaf(tree_structure, level_1_key, level_2_key, leaf_labels):
    sibling_found = False
    siblings = tree_structure[level_1_key]
    
    for sibling_key, sibling_content in siblings.items():
        if sibling_key != level_2_key and isinstance(sibling_content, dict):
            for sibling_leaf_key, sibling_leaf_labels in sibling_content.items():
                if isinstance(sibling_leaf_labels, list) and len(sibling_leaf_labels) + len(leaf_labels) <= 15:
                    sibling_leaf_labels.extend(leaf_labels)  
                    sibling_found = True
                    print(f"Merged {leaf_labels} into sibling leaf node {sibling_leaf_key} under {sibling_key}.")
                    break
            if sibling_found:
                break  

    if sibling_found and level_2_key in tree_structure[level_1_key]:
        del tree_structure[level_1_key][level_2_key]
        print(f"Deleted node {level_2_key} after merging its leaf nodes.")

    if not sibling_found:
        if level_2_key not in tree_structure[level_1_key]:
            tree_structure[level_1_key][level_2_key] = {}  
        new_leaf_key = f'Merged_Leaf_{len(tree_structure[level_1_key][level_2_key])}'
        tree_structure[level_1_key][level_2_key][new_leaf_key] = leaf_labels
        print(f"Created new leaf node {new_leaf_key} under {level_2_key} for {leaf_labels}.")

def split_leaf_node(tree_structure, level_1_key, level_2_key, level_3_key, leaf_labels):
    half = len(leaf_labels) // 2
    tree_structure[level_1_key][level_2_key][level_3_key + '_1'] = leaf_labels[:half]
    tree_structure[level_1_key][level_2_key][level_3_key + '_2'] = leaf_labels[half:]
    print(f"Split node {level_3_key} into two sub-nodes.")

tree_structure = hierarchical_clustering(category_means, n_clusters_level_1, n_clusters_level_2, n_clusters_level_3)

def train_classifiers(tree_structure, x_train, y_train, x_val, y_val):
    for level_1_key, level_2_dict in tree_structure.items():
        for level_2_key, level_3_dict in level_2_dict.items():
            if isinstance(level_3_dict, dict):
                for level_3_key, leaf_labels in level_3_dict.items():
                    train_leaf_classifier(level_3_key, leaf_labels, x_train, y_train, x_val, y_val)
            else:
                print(f"################ Processing leaf node at Level 2: {level_2_key} #####################")
                print(f"################ Processing leaf node at Level 2: {level_3_dict} #####################")
                train_leaf_classifier(level_2_key, level_3_dict, x_train, y_train, x_val, y_val)

def train_leaf_classifier(leaf_node_key, leaf_labels, x_train, y_train, x_val, y_val):
    filtered_train_indices = np.isin(y_train, leaf_labels)
    filtered_val_indices = np.isin(y_val, leaf_labels)

    if np.sum(filtered_train_indices) < 2 or np.sum(filtered_val_indices) < 2:
        print(f"################ {leaf_labels} #####################")
        print(f"################ {filtered_train_indices} #####################")
        print(f"################ {filtered_val_indices} #####################")
        print(f"Skipping leaf node {leaf_node_key} due to insufficient data.")
        
    x_train_leaf = x_train[filtered_train_indices]
    y_train_leaf = y_train[filtered_train_indices]
    x_val_leaf = x_val[filtered_val_indices]
    y_val_leaf = y_val[filtered_val_indices]

    y_train_leaf_mapped = map_labels(y_train_leaf, leaf_labels)
    y_val_leaf_mapped = map_labels(y_val_leaf, leaf_labels)

    num_classes = len(leaf_labels)
    y_train_leaf_one_hot = to_categorical(y_train_leaf_mapped, num_classes=num_classes)
    y_val_leaf_one_hot = to_categorical(y_val_leaf_mapped, num_classes=num_classes)

    if leaf_node_key == "Level_3_101_1":
        print(f"Using P1 classifier for leaf node {leaf_node_key} with classes {leaf_labels}")
        create_P1_classifier(x_train_leaf, y_train_leaf_one_hot, x_val_leaf, y_val_leaf_one_hot, num_classes)
    elif leaf_node_key == "Level_3_101_2":
        print(f"Using P1 classifier for leaf node {leaf_node_key} with classes {leaf_labels}")
        create_P1_classifier(x_train_leaf, y_train_leaf_one_hot, x_val_leaf, y_val_leaf_one_hot, num_classes)
    else:
        print(f"Using default P2 classifier for leaf node {leaf_node_key} with classes {leaf_labels}")
        create_P2_classifier(x_train_leaf, y_train_leaf_one_hot, x_val_leaf, y_val_leaf_one_hot, num_classes)

def create_P2_classifier(x_train, y_train, x_val, y_val, num_classes):
    model = Sequential()
    model.add(Embedding(256, 64, input_length=x_train.shape[1]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    plot_model(model, to_file='./model_P2.png', show_shapes=True)    
    classification_report_callback = ClassificationReportCallback(x_val, y_val, num_classes)
    
    model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[classification_report_callback])

def create_P1_classifier(x_train, y_train, x_val, y_val, num_classes):
    model = Sequential()
    model.add(Embedding(256, 64, input_length=x_train.shape[1]))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    plot_model(model, to_file='./model_P1.png', show_shapes=True)    
    classification_report_callback = ClassificationReportCallback(x_val, y_val, num_classes)

    model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[classification_report_callback])

def create_byte_level_LSTM_classifier(x_train, y_train, x_val, y_val, num_classes):
    model = Sequential()
    model.add(Embedding(256, 32, input_length=x_train.shape[1]))    
    model.add(LSTM(64, return_sequences=True))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(LSTM(32))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))    
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.3))    
    model.add(Dense(num_classes, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])  
    classification_report_callback = ClassificationReportCallback(x_val, y_val, num_classes)

    model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), batch_size=64, callbacks=[classification_report_callback])
    return model

def create_CNN_LSTM_classifier(x_train, y_train, x_val, y_val, num_classes):
    model = Sequential()   
    model.add(Embedding(256, 16, input_length=x_train.shape[1]))
    model.add(Conv1D(filters=8, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))  
    model.add(Conv1D(filters=8, kernel_size=3, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPool1D(2))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    classification_report_callback = ClassificationReportCallback(x_val, y_val, num_classes)

    model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_val, y_val), callbacks=[classification_report_callback])
    return model

def visualize_tree(tree):
    dot = Digraph()
    
    def add_edges(tree, parent_name):
        for key, value in tree.items():
            node_name = str(key)
            dot.node(node_name, node_name)  
            dot.edge(parent_name, node_name)  
            if isinstance(value, dict):
                add_edges(value, node_name)  
    
    dot.node('Root', 'Root')  
    add_edges(tree, 'Root')  
    return dot

tree_visualization = visualize_tree(tree_structure)

output_file = 'tree_structure'  
output_format = 'png'  

tree_visualization.render(output_file, format=output_format, cleanup=True)  
print(f"树形结构图已保存为 {output_file}.{output_format}")

train_classifiers(tree_structure, x_train, y_train, x_val, y_val)
