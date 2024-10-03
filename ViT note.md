# Vision Transformers (ViT) - Class Notes

## Introduction

Transformers have revolutionized natural language processing by introducing mechanisms that allow models to focus on different parts of the input sequence when generating outputs. The **Vision Transformer (ViT)** extends this architecture to handle image data, offering an alternative to convolutional neural networks (CNNs) for image classification tasks.

**Key Paper:** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Google Research, 2020)

---

## ViT Architecture

### Overview

- **Encoder-Only Transformer:** ViT uses only the encoder part of the Transformer architecture, omitting the decoder used in sequence generation tasks.
- **Input Representation:** Images are converted into a sequence of tokens to mimic the sequential nature of language models.

### Input Embeddings

#### Image Patching

- **Process:**
  - An input image is divided into fixed-size patches (e.g., 16×16 pixels).
  - Each patch is flattened into a vector.
- **Purpose:** Transforms a 2D image into a sequence of patches, enabling the use of Transformer models designed for sequential data.

#### Linear Projection of Patches

- **Embedding Generation:**
  - Each flattened patch vector is passed through a linear projection (fully connected layer).
  - The output is an embedding vector representing the patch in a high-dimensional space.
- **Benefit:** Allows the model to learn rich representations of image patches.

### CLS Token (Classification Token)

- **Definition:** A learnable embedding vector prepended to the sequence of patch embeddings.
- **Purpose:**
  - Acts as an aggregate representation of the entire image.
  - After processing through the Transformer encoder, the CLS token contains information for classification tasks.
- **Analogy:** Similar to using a special token in NLP tasks that gathers information from the entire sequence.

### Positional Embeddings

- **Need for Positional Information:**
  - Transformers are position-invariant; they need explicit positional information to understand the order or spatial relationships.
- **Implementation:**
  - Learnable positional embeddings are added to each patch embedding and the CLS token.
- **Benefit:** Provides spatial context, allowing the model to understand the position of patches within the image.

---

## Transformer Encoder Components

The ViT uses standard Transformer encoder blocks, comprising several key components.

### Multi-Head Attention

- **Mechanism:**
  - Allows the model to focus on different positions within the input sequence simultaneously.
  - Consists of multiple attention "heads" that learn different aspects of the input.
- **Queries, Keys, Values:**
  - **Queries (Q):** Derived from the input embeddings; represent the current position.
  - **Keys (K):** Also derived from the input embeddings; represent the positions to compare against.
  - **Values (V):** Carry the actual information to be aggregated.
- **Scaled Dot-Product Attention:**
  - Attention weights are computed using the dot product of queries and keys, scaled, and passed through a softmax function.
  - The output is a weighted sum of the values.

### Layer Normalization

- **Purpose:** Stabilizes and accelerates training by normalizing inputs across the features for each sample.
- **Placement in ViT:**
  - Applied before (Pre-Norm) or after (Post-Norm) the attention and feedforward layers.
- **Why Layer Norm over Batch Norm:**
  - Layer Norm operates on individual samples, making it suitable for variable-length sequences and small batch sizes common in Transformer models.

### Feedforward Network

- **Structure:**
  - A two-layer fully connected network with an activation function (commonly GELU).
- **Function:**
  - Applies non-linear transformations to each position separately and identically.
- **Dropout and Activation:**
  - Includes dropout layers to prevent overfitting.
  - Activation functions like GELU (Gaussian Error Linear Unit) introduce non-linearity.

### Residual Connections (Skip Connections)

- **Purpose:**
  - Helps with gradient flow during backpropagation.
  - Allows the model to learn identity functions, facilitating deeper architectures.
- **Implementation:**
  - The input to a sub-layer (e.g., attention, feedforward) is added to its output.
  - Often combined with layer normalization (Pre-Norm or Post-Norm setups).

---

## Implementing ViT

### Efficient Image Patching with Einops

- **Einops Library:**
  - Provides flexible tensor operations for rearranging and reshaping multi-dimensional data.
- **Patching Implementation:**
  - Rearranges the image tensor to extract patches efficiently.
  - For example, an image tensor of shape `(batch_size, channels, height, width)` can be reshaped to extract patches of size `(patch_size, patch_size)`.

#### Code Snippets for Patching (Simplified)

```python
import einops

# Example: Extract patches
patches = einops.rearrange(
    images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
    p1=patch_size, p2=patch_size
)
```

### Transformer Encoder Components Implementation

#### Multi-Head Attention

- Utilize PyTorch's `nn.MultiheadAttention` module.
- Linear layers to project inputs to queries, keys, and values.

#### Layer Normalization

- Implemented using `nn.LayerNorm`.
- Wrapped around sub-layers using a `PreNorm` module.

#### Feedforward Network

- Two linear layers with an activation function (e.g., GELU).
- Optional dropout layers for regularization.

#### Residual Connections

- Combine inputs and outputs of sub-layers using addition.
- Implemented within a `Residual` module for cleaner code.

### Putting It All Together

#### ViT Model Structure

- Input images are patched and embedded.
- CLS token and positional embeddings are added.
- The sequence is passed through multiple Transformer encoder layers.
- The output corresponding to the CLS token is used for classification.

---

## Comparison Between CNNs and ViT

### Inductive Biases

- **CNNs:**
  - Strong inductive biases like translation invariance and locality due to convolutional kernels.
  - Effective with smaller datasets due to these built-in assumptions.
- **ViTs:**
  - Fewer inductive biases, relying more on the data to learn patterns.
  - Requires larger datasets to perform well.

### Data Requirements

- **CNNs:**
  - Can perform well on smaller datasets.
  - The architectural biases help generalize from fewer examples.
- **ViTs:**
  - Perform better with large-scale datasets (e.g., millions of images).
  - Without strong inductive biases, they need more data to learn effectively.

### Hierarchical Learning

- **CNNs:**
  - Learn hierarchical representations.
  - Early layers capture local features; deeper layers capture global patterns.
- **ViTs:**
  - Use global self-attention, allowing each patch to attend to all others from the beginning.
  - May lack the hierarchical feature learning of CNNs.

### Global vs. Local Processing

- **CNNs:**
  - Local receptive fields process information locally before combining at higher levels.
- **ViTs:**
  - Global attention allows modeling of long-range dependencies from the start.
  - Potentially captures global context more effectively.

---

## ViT Variants

### Swin Transformer (Shifted Window Transformer)

- **Objective:** Introduce hierarchical feature learning into Transformers.
- **Key Concepts:**
  - **Window-Based Attention:**
    - Apply self-attention within local windows, reducing computational complexity.
  - **Shifted Windows:**
    - Alternate between regular and shifted window partitions between layers.
    - Allows cross-window connections, enabling information flow between windows.
  - **Hierarchical Representation:**
    - Patches are merged progressively, reducing the number of tokens and increasing the receptive field.
- **Benefits:**
  - Combines the strengths of CNNs (hierarchical representations) with Transformers (global modeling).

### Data-Efficient Image Transformer (DeiT)

- **Objective:** Improve ViT performance on smaller datasets.
- **Key Concepts:**
  - **Knowledge Distillation:**
    - A CNN teacher model guides the ViT student model.
    - Helps the ViT learn inductive biases inherent in CNNs.
  - **Distillation Token:**
    - An additional token similar to the CLS token.
    - Specifically learns to match the teacher's predictions.
- **Benefits:**
  - Enhances ViT's performance when data is limited.
  - Combines strengths of both architectures.

---

## Applications and Interpretability

### Attention Maps

- **Interpretability:**
  - Attention weights can be visualized to understand which parts of the image the model focuses on.
- **Attention Map Visualization:**
  - Helps in analyzing the model's decision-making process.
  - Useful for tasks requiring explainability, such as medical imaging.

---

## Conclusion

The Vision Transformer (ViT) represents a significant shift in applying Transformer architectures to image data. By treating images as sequences of patches, ViT leverages the powerful self-attention mechanisms of Transformers to model global relationships within images.

**Key Takeaways:**

- ViT extends the Transformer encoder to images by patching and embedding images into sequences.
- The model relies on positional embeddings and a classification token to process images effectively.
- ViT performs best with large datasets due to fewer inductive biases compared to CNNs.
- Variants like Swin Transformer and DeiT aim to combine the strengths of CNNs and Transformers, addressing data efficiency and hierarchical feature learning.

---

## Potential Interview Topics

- **Understanding of Transformer Architecture:**
  - Explain self-attention, multi-head attention, and the role of queries, keys, and values.
- **ViT Specifics:**
  - How images are transformed into sequences of patches.
  - The purpose and implementation of the CLS token.
- **Positional Embeddings:**
  - Why positional information is essential in Transformers and how it's added in ViT.
- **Comparison with CNNs:**
  - Discuss inductive biases, data requirements, and hierarchical learning.
- **Variants of ViT:**
  - Describe the Swin Transformer and DeiT architectures.
  - Explain how they address some of the limitations of the original ViT.
- **Implementing ViT:**
  - Discuss how to efficiently implement image patching (e.g., using Einops).
  - Explain the components of the Transformer encoder in the context of ViT.
- **Interpretability:**
  - How attention maps can be used to understand model predictions.

---

https://www.youtube.com/@umarjamilai/featured
https://www.youtube.com/@mildlyoverfitted/playlists
https://github.com/pinecone-io/examples/blob/master/learn/search/image/image-retrieval-ebook/vision-transformers/vit.ipynb
https://colab.research.google.com/drive/1P9TPRWsDdqJC6IvOxjG2_3QlgCt59P0w?usp=sharing#scrollTo=V7LkzG6fOt_k
