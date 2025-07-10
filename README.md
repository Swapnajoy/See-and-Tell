# See-and-Tell
A Vision-Language Model with Retrieval-Augmented Generation (VLM-RAG).

## 📜 Project Overview
This project explores **retrieval-augmented generation (RAG)**, specifically for vision-language tasks, where the model generates captions for images by leveraging both the image content and relevant textual information retrieved from an external knowledge base.

The **T5** model is used as the **text encoder**, which processes the query text and generates the final caption. The **ViT (Vision Transformer)** is used as the **image encoder**, which extracts meaningful features from the image. These embeddings from both the text and image are then fused with additional information from a retrieval system, making the model more contextually aware.

### 🎯 Purpose
The goal of this project was to enhance traditional image captioning by integrating retrieval-augmented generation. By using external knowledge through retrieval, it was aimed to **generate more descriptive captions** for the image especially the description of the objects in the image.

Through this project, I learned several key concepts:
- **Text-to-text transformers (T5)** are extremely effective at generating coherent text from a variety of inputs.
- **Vision Transformers (ViT)** provide powerful image representations, which can be directly leveraged for multimodal tasks.
- **Retrieval-augmented generation** is a powerful technique for improving captioning and other language generation tasks, where external information is needed.
- Fine-tuning and training complex models with **multimodal inputs** and retrieval components require careful handling of overfitting, data diversity, and proper training strategies.

This project served as both a practical exercise in working with **transformer-based models** and a deep dive into the **retrieval-augmented generation** paradigm.

## 🌍 Data Creation

The project relies heavily on the **COCO 2017 Validation Split** dataset and external knowledge from **Wikipedia**. Here's how the data was processed and used:

- **COCO Dataset with Annotations:** Utilized images and their corresponding captions from the **COCO dataset** to train and evaluate our model. The dataset includes detailed captions describing the objects and scenes in the images.
- **Noun Extraction:** Performed **noun extraction** using **SpaCy** to identify the important objects and concepts in the captions.
- **Wiki Search for Knowledge Base:** For each noun identified in the captions, searched for relevant information on **Wikipedia** to create a **knowledge base**. This additional context helps enhance the model's understanding of objects in the images.
- **Sentence Embedding:** Used **SentenceTransformer** to encode the text from the knowledge base into embeddings. These embeddings are used to find similar content during training.
- **Faiss Indexing:** The embeddings of the knowledge base content were indexed using **Faiss** to enable fast and efficient retrieval during both training and inference.

These steps resulted in a rich dataset with textual information related to the images, improving the model's ability to understand and generate detailed descriptions.



