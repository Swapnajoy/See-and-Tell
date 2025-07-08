# See-and-Tell
A Vision-Language Model with Retrieval-Augmented Generation (VLM-RAG).

## Project Overview
This project explores **retrieval-augmented generation (RAG)**, specifically for vision-language tasks, where the model generates captions for images by leveraging both the image content and relevant textual information retrieved from an external knowledge base.

The **T5** model is used as the **text encoder**, which processes the query text and generates the final caption. The **ViT (Vision Transformer)** is used as the **image encoder**, which extracts meaningful features from the image. These embeddings from both the text and image are then fused with additional information from a retrieval system, making the model more contextually aware.

Key features of this implementation:
- **Image and text encoding**: The image is encoded using **ViT**, and the text is encoded using **T5**.
- **Retrieval-enhanced captioning**: Uses a retrieval-based mechanism to inject relevant external knowledge into captions.
- **Learnable fusion weights**: The fusion of text, image, and retrieval vectors is controlled by learnable parameters.
- **Robust training**: Techniques such as dropout and distractor retrieval are employed to prevent overfitting and improve generalization.

