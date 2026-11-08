# Cephalometric Landmark Loss

A geometry-aware loss for cephalometric landmark detection, implemented in PyTorch.

## The idea

Cephalometric analysis places 19 landmarks on a skull X-ray and measures the distances and angles between them. A model that predicts one heatmap per landmark is usually trained with plain MSE, which treats every landmark on its own and ignores whether the predicted landmarks keep the right geometry relative to each other.

This loss multiplies the heatmap MSE by an **anatomical-context weight**:

$$\omega_{ac}(h^c, o^c) = \log_2 \lVert A(h^c) - A(o^c) \rVert_2^2 + \log_2 \lVert D(h^c) - D(o^c) \rVert_2^2$$

where $A(\cdot)$ and $D(\cdot)$ are the 19×19 matrices of angles and distances between every pair of landmarks, for the ground truth $h^c$ and the prediction $o^c$.

## How it's computed

1. Take each landmark's position from the peak of its heatmap.
2. Build the pairwise distance matrix and the pairwise angle matrix, for both prediction and ground truth.
3. Take the log₂ of the Euclidean difference between the two distance matrices, and the same for the two angle matrices. Their sum is the weight.
4. Multiply the heatmap MSE by that weight.

The network is an encoder–decoder that takes a 1-channel 512×512 image and outputs 19 heatmaps.

## What the notebook shows

- The loss tested on synthetic data before training.
- A training loop with Adam.
- The gradient norm of every layer, to confirm gradients reach the whole network.

The first version of the loss returned −inf (a log of zero), which blocks backpropagation, so every sub-function was checked line by line.

## Limits and next steps

- It runs on **synthetic data only**. It hasn't been trained on real cephalograms.
- Landmark positions come from a hard argmax, and the weight is converted to a plain number before it multiplies the MSE. So the geometry term scales the loss but doesn't push landmarks into place by itself. A soft-argmax, with the weight kept as a tensor, would make the geometry term fully differentiable.

## Author

Abdelrahman Ibrahim · [GitHub profile](https://github.com/moh246) · [LinkedIn](https://www.linkedin.com/in/abdelrahman-mohsen-4438991a8/)
