Link to the whole project: https://uoe-my.sharepoint.com/:f:/g/personal/s2340198_ed_ac_uk/EnTRwKm_d9JGiSum8Uf2aDwBZLm3z3W7ofbGxJmdR56IYA?e=YZjJdI

# Data Preprocessing and EDA

## Sartaj's Notebooks

### 1. Investor-Startup Relationship Data Processing
- **Notebook:** `investor_startup_relation_dataprocess.ipynb`
- **Required files:** organizations.csv, investments.csv, funding_rounds.csv, investors.csv
- **Description:** Defines the relationship between startups and investors.
- **Output files:** investor_startup.csv, investor_startup_rel.csv, investor_startup_rel_freq.csv

### 2. Exploratory Data Analysis (EDA)
- **Notebook:** `EDA.ipynb`
- **Required files:** investor_startup.csv
- **Description:** Contains EDA done by Sartaj, as mentioned in his thesis.
- **Output files:** EDA images

### 3. Co-Investors Data Processing
- **Notebook:** `co-investors_dataprocessing.ipynb`
- **Required files:** investments.csv
- **Description:** Finds the number of times two investors invested together.
- **Output files:** co_investors.csv

### 4. Lead Investor Data Processing
- **Notebook:** `lead_investor_dataprocess.ipynb`
- **Required files:** investor_startup_rel.csv
- **Description:** Finds the number of times one investor was the lead investor of another.
- **Output files:** lead_investors.csv

### 5. Investor Text Embedding
- **Notebook:** `investor_text_embedding.ipynb`
- **Required files:** organizations.csv, organization_descriptions.csv, investor_startup_rel.csv, people_descriptions.csv
- **Description:** Embeds investor descriptions into feature vectors.
- **Output files:** investor_text_embedding.csv

### 6. Investor Features (Preferences and Attributes)
- **Notebook:** `Investor_features(preferences and attributes).ipynb`
- **Required files:** organizations.csv, investments.csv, investor_startup.csv, investors.csv, investor_text_embedding.csv
- **Description:** Finds and encodes investor features.
- **Output files:** investor_features_n_embedding.csv

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Zain's Notebooks

### 1. Startup Features Data Processing
- **Notebook:** `startup_features_dataprocess.ipynb`
- **Required files:** investor_startup.csv
- **Description:** Finds and encodes startup features.
- **Output files:** startup_features.csv (last one used to merge with text embedding)

### 2. Exploratory Data Analysis (EDA) - Final
- **Notebook:** `EDA_final.ipynb`
- **Required files:** investor_startup.csv
- **Description:** Contains EDA done by Zain, as mentioned in his thesis.
- **Output files:** EDA images

### 3. Startup Text Embedding
- **Notebook:** `startup_text_embed.ipynb`
- **Required files:** organizations.csv, organization_descriptions.csv, investor_startup_rel.csv, startup_features.csv
- **Description:** Embeds startup descriptions into feature vectors.
- **Output files:** startup_features_n_text_embedding.csv

### 4. Rivalry Scores - Final
- **Notebook:** `rivalry_scores_final.ipynb`
- **Required files:** investor_startup.csv
- **Description:** Finds rivalry between two investors.
- **Output files:** rivalry_scores.csv


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Graphsage + KNN

## Project Workflow

### 1. Graph Creation
- **Script**: `graph_creation.py`
- **Prerequisites**: Python Libraries: pandas, warnings, torch, torch_geometric, json, argparse, logging, os, sklearn
- **Description**: Process data to generate graph representations.
- **Usage**:
    - python graph_creation_new.py [options]
    - Options:
        - --verbose: Use this option to display messages.
        - --ofile: Default is "data.pt".
        - --ofile_df: Default is "test_startups_r.csv".
        - --ofile_in: Default is "investor_name_dict.json".
        - --ofile_sn: Default is "startup_name_dict.json".
        - --ofile_sm: Default is "startup_map.json".
- **Output files:** data.pt, train_for_knn.csv, test_startups_r.csv, Various JSON mapping files for investors and startups


### 2. KNN Graph Generation
- **Notebook**: `KNN_startups.ipynb`
- **Prerequisites**: Python Libraries: torch, torch_geometric, pandas
- **Description**: Generates KNN graphs. Utilizes `train_startups` from the graph creation step.
- **Output files:**  data_st_des_fet_r.pt

### 3. Model
- **Script**: `model.py`
- **Prerequisites**: Python Libraries: warnings, torch, torch_geometric
- **Description**: The script defines neural network models, particularly graph neural network layers.

### 4. Model Training with Optuna
- **Script**: `train_optuna.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, matplotlib, warnings, torch, torch_geometric, sklearn, tqdm, optuna, numpy
- **Description**: Train the graph neural network model with different hyperparameters using optuna for finding optimized hyperparameters with data and graphs from previous steps.
- **Usage**:
    - python train.py [options]
    - Options:
        - --ifile: Input file. Default is "data.pt".
        - --lr: Learning rate. Default is "1e-04".
        - --weight_decay: Weight decay. Default is "1e-05".
        - --batch_size: Batch size. Default is "128".
        - --odir: Output directory. Default is "None".
        - --model_path: Path to save the trained model. Default is "gsage.pt".
- **Output files:** Multiple Models, results.txt file


### 4. Model Training
- **Script**: `train.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, matplotlib, warnings, torch, torch_geometric, sklearn, tqdm
- **Description**: Train the graph neural network model using the best hyperparmeters only.
- **Usage**:
    - python train.py [options]
    - Options:
        - --ifile: Input file. Default is "data.pt".
        - --lr: Learning rate. Default is "1e-04".
        - --weight_decay: Weight decay. Default is "1e-05".
        - --batch_size: Batch size. Default is "128".
        - --odir: Output directory. Default is "None".
        - --model_path: Path to save the trained model. Default is "gsage.pt".
- **Output files:** model_gsageemb_and_fea__1e-04_1e-05_128.pth(Best Model)


### 5. Inference
- **Script**: `inference.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, warnings, torch, torch_geometric, pandas, numpy, json
- **Data Required**: data.pt, data_st_des_fet_r.pt, investor_name_dict, startup_name_dict
- **Description**: Perform inference or predictions using the trained model.
- **Usage**:
    - python train.py [options]
    - Options:
        - --ifile: Input file. Default is "data.pt".
        - --ofile_df: Output file for dataframe. Default is "test_startups_r.csv".
        - --ofile_in: Output file for investor names. Default is "investor_name_dict.json".
        - --ofile_sn: Output file for startup names. Default is "startup_name_dict.json".
        - --ofile_sm: Output file for startup map. Default is "startup_map.json".
        - --ofile_im: Output file for investor map. Default is "investor_map.json".
        - --odir: Output directory. Default is "None".
        - --model_path: Path to the trained model. Default is "model_gsageemb_and_fea__1e-04_1e-05_128.pth".

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Gat2conv

## 1. Write Edge Script
- **Script**: `write_edge.py`
- **Description**: Processes and transforms raw investor-startup relationship data into a suitable format. Generates edge indices for a graph representation of the data.

### Prerequisites
- **Python Libraries**: pandas, numpy, torch, os, argparse, logging
- **Data**:
  - `investor_startup_rel_freq.csv`: Relationships between investors and startups.
  - `investor_features_n_embedding.csv`: Features and embeddings for investors.
  - `startup_features_n_text_embedding.csv`: Features and text embeddings for startups.

### Usage
python write_edge.py [options]
- **options**
    - `--verbose`: Display debugging messages.
    - `--ofile`: Output file name (default: `data.npz`).
    - `--odir`: Output directory (created if non-existent).

### Outputs
- `data.npz`: Contains `x0` (investor features), `x1` (startup features), and `edge_index` (edge indices from investors to startups).

---

## 2. PCA Transformation Script
- **Script**: `dim_reduction.py`
- **Description**: Processes and transforms raw feature data using PCA for dimensionality reduction.


### Prerequisites
- **Python Libraries**: pandas, numpy, os, argparse, logging, pickle, sklearn
- **Data**:
  - `data.npz`: Raw feature data for investors (`x0`) and startups (`x1`).


### Usage
python pca_transform.py [options]
- **options**
    - `--verbose`: Display debugging messages.
    - `--ifile`: Input file name.
    - `--k`: Number of principal components (default: 512).
    - `--n_sample`: PCA sample number (default: 25000).
    - `--ofile`: Output file name.
    - `--odir`: Output directory (created if non-existent).


### Outputs
- `None.npz`: PCA-transformed features for investors (`x0`) and startups (`x1`).
- `pca_model.pkl`: Saved PCA model.
- Various `.npy` files: Statistical measures like variance mask and data mean.

---

## 3. Graph Neural Network Layers
- **Script**: `layers.py`
- **Description**: Implementation of the Gat and Model layers for predicting investor-organization connections.

### Prerequisites
- **Python Libraries**: torch, torch.nn, torch.nn.functional, torch_geometric.nn
- **Data**:
  - `data.npz`: Raw feature data for investors (`x0`) and startups (`x1`).

### Classes
- **GCN**
    - **Description**:GCN layer based on GATv2Conv from PyTorch Geometric with multi-head attention.
    - **Parameters**:
        - n_layers: Number of GCN layers.
        - in_dim: Input feature dimension.
        - n_heads: Number of attention heads in GATv2Conv.
        - Forward Method: Takes in node features, edge indices, and edge attributes and returns the updated node features after passing through the GCN layers.
 **Note**: Import `layers.py` into the main or training script for using the provided GCN and Model layers.


- **Model**
    - **Description**:Main model predicting connections between investors and organizations using the GCN layer.
    - **Parameters**:
        - k: Number of nearest neighbors.
        - in_dim: Dimension of input node features.
        - graph_dim: Dimension of graph embeddings.
        - Forward Method: Takes in node features, edge indices, edge attributes, and counts of investor and unknown nodes. Returns a matrix representing the predicted connections between investors and organizations.
**Note**: This script, layers.py, should be imported into your main script or training script to utilize the provided GCN and Model layers.

### 4. Model Training with Optuna
- **Script**: `train_optuna_tuning.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, matplotlib, warnings, torch, torch_geometric, sklearn, tqdm, optuna, numpy
- **Description**: Train the graph neural network model with different hyperparameters using optuna for finding optimized hyperparameters with data and graphs from previous steps.

- **Output files:** Multiple Models, results.txt file


### 5. Model Training
- **Script**: `train.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, matplotlib, warnings, torch, torch_geometric, sklearn, tqdm
- **Description**: Train the graph neural network model using the best hyperparmeters only.
- **Usage**:
    - python train.py [options]
    - Options:
        - --ifile: Input file. Default is "None.npz".
        - --lr: Learning rate. Default is "0.00012156299936369672".
        - --weight_decay: Weight decay. Default is "0.0021852322399311197".
        - --k, default = "17"
        - --batch_size: Batch size. Default is "12".
        - --odir: Output directory. Default is "None".
        - --model_path: Path to save the trained model. Default is "model_best_check.pth".
- **Output files:** model_best_check.pth(Best Model)


### 6. Inference
- **Script**: `Inference_new_node.py`
- **Prerequisites**: Python Libraries: os, argparse, logging, warnings, torch, torch_geometric, pandas, numpy, json
- **Data Required**: pca_model.pkl, variance_mask_x1.npy, Xm_x1.npy, mean_pca_x1.npy, std_pca_x1.npy, investor_name_dict, startup_name_dict, test_startups_r.csv
- **Description**: Perform inference or predictions using the trained model.
