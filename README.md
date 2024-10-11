# Uncertainty Quantification on GNN using Stochastic Centering

This project focuses on implementing Uncertainty Quantification for Graph Neural Networks (GNNs) using Stochastic Centering techniques. The approach aims to provide reliable uncertainty estimates for GNN predictions, which is crucial in tasks such as node classification, link prediction, and graph classification.

## Setup

### 1. Install Dependencies

- Install the required dependencies by running:
    ```bash
    poetry install
    ```

### 2. Configure SLURM (Optional)

- If you are using a GPU cluster with SLURM for job scheduling, adjust the SLURM settings in the configuration file:
    ```
    config/seml/<your-experiment>.yaml
    ```
  Update the SLURM parameters such as `partition`, `gpus`, and `cpus-per-task` according to your cluster's configuration.

### 3. Run the Experiment

- Execute the experiment script to start training and evaluation:
    ```bash
    python experiment_<your-experiment>.py
    ```
  Replace `<set-up>` with the appropriate script name, such as `experiment_baseline.py` or `experiment_advanced.py`.
