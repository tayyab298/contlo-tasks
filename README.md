# contlo-tasks
# GPT-2 Custom Model Implementation

This repository contains an implementation of a custom GPT-2 model using PyTorch. The model consists of various components such as self-attention mechanisms, layer normalization, and feedforward layers.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Usage](#usage)
- [Sample Usage](#sample-usage)
- [Hyperparameters](#hyperparameters)

## Overview

This codebase implements a custom GPT-2 architecture using PyTorch. It includes:

- `GPT2LayerNorm`: Layer normalization module used in the GPT-2 architecture.
- `MultiHeadSelfAttention`: Implements multi-head self-attention mechanism.
- `PositionWiseFeedForward`: Module for position-wise feedforward layers.
- `GPT2Block`: A block comprising self-attention, normalization, and feedforward layers.
- `GPT2OutputLayer`: Output layer for the GPT-2 model.
- `GPT2`: Main GPT-2 model architecture incorporating multiple blocks and output layer.

## Components

### GPT2LayerNorm

This class implements layer normalization used within the GPT-2 architecture.

### MultiHeadSelfAttention

Implements multi-head self-attention mechanism with query, key, and value mappings.

### PositionWiseFeedForward

Position-wise feedforward module consisting of linear layers.

### GPT2Block

A block within the GPT-2 architecture, including attention, normalization, and feedforward layers.

### GPT2OutputLayer

Output layer for the GPT-2 model mapping embeddings to vocabulary.

## Usage

The primary usage includes initializing and utilizing the `GPT2` class:

```python
# Create a GPT-2 model instance
gpt2_model = GPT2(vocab_size, embed_size, num_heads, hidden_size, num_layers)

# Input tensor for sequence generation
input_ids = torch.tensor([[2, 15, 22, 45, 3, 0]])

# Get model output for input sequence
output = gpt2_model(input_ids)
print(output.shape)  # Adjust based on actual output shape

# Sample Usage
The generate method within the GPT2 class facilitates sequence generation:

# Generate sequences using the GPT-2 model
generated_sequences = gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

Hyperparameters
The GPT-2 model is initialized with the following hyperparameters:

vocab_size: Vocabulary size.
embed_size: Embedding size.
num_heads: Number of attention heads.
hidden_size: Hidden layer size.
num_layers: Number of transformer blocks.

```

Task 2
Architecture
Components
SlidingWindowAttention: Implements attention mechanism with sliding window functionality.
GroupQueryAttention: Implements attention by grouping queries.
RotaryPositionalEncoding: Generates positional encodings using rotary embeddings.
TransformerBlock: A block comprising attention and feedforward layers.
GPT2: GPT-2 model architecture with configurable attention mechanisms.
