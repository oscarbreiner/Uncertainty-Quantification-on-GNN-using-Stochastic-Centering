# Uncertainty Quantification on GNN using Stochastic Centering

Collaborators: Oscar Breiner, Constantin von Witzleben, Paul Ungermann

The goal of our project was to evaluate stochastic centering as a method for uncertainty quantification for GNNs. A recent paper investigated this concept and termed it G-ΔUQ. The authors of G-ΔUQ claim that the model is an implicit ensemble model due to the stochasticity of the anchor nodes. However, the authors of G-ΔUQ only provided empirical evidence primarily through the empirical graph NTK. Furthermore, the authors mostly benchmarked their method on calibration. They did not analyze the behavior on node classification tasks.

Therefore, we implemented and evaluated G-ΔUQ on a node classification task using different test time distribution shifts. We used a feature perturbation shift and a leave-out-class distribution shift. We compared G-Δ-UQ to a dropout model and an ensemble model. To enable a fair comparison, we used the same backbones and hyperparameters for all models. We found that G-ΔUQ performs fairly even to ensemble models. This aligns with the theory because G-ΔUQ is an implicit ensemble. However, G-ΔUQ completely ignores the non-iid nature of the data. Because of that, we developed new, more graph-structure-aware methods to enhance G-ΔUQ's performance. Unfortunately, our extensions did not lead to a significant improvement.

Conclusively, our empirical findings align with the claim, that G-ΔUQ is an implicit ensemble using only one model without compromising performance. While initial attempts at making the method graph-structure-aware showed no performance improvement, further methods could be tested, specifically finding a balanced focus between local and global graph properties.

# Seed Paper

- Single Model Uncertainty Estimation via Stochastic Data Centering: (https://arxiv.org/abs/2207.07235)
- A Survey of Uncertainty in Deep Neural Networks: (https://arxiv.org/abs/2107.03342)
- Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks: (https://arxiv.org/abs/2309.10976)



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
