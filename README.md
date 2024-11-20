# League Draft Analyzer Neural Network (LDANet)

This document provides a comprehensive overview of the LDANet deep learning model used to analyze League of Legends drafts and predict match outcomes. The focus is on explaining the network architecture, input features, data processing, training methodology, and reasoning behind design decisions.

---

## First Note

The LDANet model tackles a uniquely challenging problem: predicting match outcomes SOLELY based on the **draft phase**, without considering team skill levels or player performance. This is especially difficult because in League of Legends, a "bad" team with a "good" draft can still lose to a "good" team with a "bad" draft due to individual skill differences.

With this strucutre, the model assumes both teams play "perfectly" or at the same level. And focuses purely on the strategic implications of champion selection to decide which **draft** is objectively better.

## A Funny Insight
During early iterations, the model consistently predicted that the **blue side** would win every game. This wasn't a bug, it was a quite a funny outcome!

The model independently learned that blue side has certain inherent advantages in League of Legends drafting, which is something pro-players and analysts already recognize.

---


## Overview

The LDANet is designed to predict the winner of a League of Legends match based on:
- Champion **picks** and **bans**
- Pre-computed **synergy** and **counter values** between champions
- The **patch** the game was played on
- Other game-related metadata

This neural network uses embeddings, attention mechanisms, and residual connections to process and learn from these inputs.

---

## Features Processed

The network processes the following features:
1. **Champion Picks**:
   - 10 champion picks (5 blue side, 5 red side) are embedded using a shared embedding layer.
   - Role-specific embeddings are also used for additional context.
   
2. **Champion Bans**:
   - Up to 10 bans (5 per team) are embedded similarly to picks.

3. **Synergy Values**:
   - Calculated as win-rate-based scores for all role-specific champion pairings.
   - Inputted into a multi-head attention layer.

4. **Counter Values**:
   - Calculated as the likelihood of one champion outperforming another in a specific matchup.
   - Provides information about individual champion strengths and weaknesses.

5. **Patch Data**:
   - Normalized numeric feature representing the game patch (e.g., patch 13.12 is represented as `1.312`).

---

## Network Architecture

### Input Layer

The input size of the network is dynamically computed based on the features to be processed. The total input size is the sum of:
- Embedding dimensions for picks and bans.
- Flattened synergy and counter values.
- Numeric patch data.

### Embedding Layers

The champion IDs are passed through embedding layers to convert them into continuous feature vectors:
- **Shared Embedding**:
  - A shared `nn.Embedding` layer processes champion IDs into a vector space of size 12.
- **Role-Specific Embeddings**:
  - Separate embeddings are used for Top, Jungle, Mid, ADC, and Support roles. This allows the network to learn role-specific relationships and dynamics.

### Attention Layers

Attention layers are applied to enhance the understanding of relationships between champions:
- **Pick Attention**:
  - Processes the role-specific embeddings of champion picks using multi-head attention.
  - Helps the network learn interdependencies between roles and champions.
- **Synergy Attention**:
  - Processes synergy values for both teams to highlight critical pairings.

### Residual Connections

Residual connections are implemented between certain layers to retain lower-level information across deeper layers. This mitigates the vanishing gradient problem and aids in faster convergence.

### Fully Connected Layers

After embeddings and attention layers, the data flows through a series of fully connected layers:
1. **Input Layer**: Takes the concatenated embeddings and other processed features.
2. **Hidden Layers**:
   - `fc1`: Input → 1024 units
   - `fc2`: 1024 → 512 units
   - `fc3`: 512 → 256 units
   - `fc4`: 256 → 128 units
   - `fc5`: 128 → 32 units
   - Each layer uses LeakyReLU activation and dropout for regularization.
3. **Output Layer**:
   - Single neuron with a sigmoid activation for binary classification (blue win vs. red win).

---

## Why This Architecture?

1. **Embeddings**:
   - Champion embeddings allow the network to generalize across champions by learning their latent properties.
   - Role-specific embeddings add contextual understanding of how champions interact within specific roles.

2. **Attention Layers**:
   - Focus on critical champion interactions and synergies.
   - Multi-head attention captures multiple perspectives of relationships (e.g., Top-Jungle synergy vs. ADC-Support synergy).

3. **Residual Connections**:
   - Prevent loss of information across layers.
   - Aid in training deeper networks by bypassing non-linear transformations.

4. **Dropout Regularization**:
   - Prevents overfitting by randomly zeroing activations during training.

5. **Fully Connected Layers**:
   - Extract high-level abstractions from the processed features.
   - Gradually reduce the dimensionality to condense information for the final prediction.

---

## Training Process

1. **Dataset Preparation**:
   - Input features are normalized to the [0, 1] range.

2. **Training Configuration**:
   - Optimizer: Adam with weight decay (`0.0001`) for regularization.
   - Loss Function: Binary Cross-Entropy Loss (BCE).
   - Learning Rate Scheduler: StepLR reduces the learning rate after a few epochs of stagnation.

3. **Metrics**:
   - Training and validation losses.
   - Validation accuracy.
   - Distribution of predictions to ensure the network predicts both classes.

---

## Key Challenges and Solutions

1. **Model Stagnation**:
   - Residual connections and attention mechanisms were added to improve gradient flow and focus on key interactions.

2. **Accuracy Plateau**:
   - Addressed by fine-tuning learning rate schedules, balancing classes, and adding contextual features like counter values.

---

## Results

- **Validation Accuracy**: Reached a peak of 70% after fine-tuning the architecture and training process.
- **Key Insights**:
  - Role-specific embeddings and attention mechanisms significantly improved performance.
  - Synergy and counter values provided crucial context for predictions.

---

*a project by lipeeeee*
