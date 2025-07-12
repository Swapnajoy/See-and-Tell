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

## 🚀 Model Architecture

In this section, we’ll describe the key components of the **VLMRAG** model. The architecture is designed to handle both **image** and **text** inputs, retrieve relevant content from an external knowledge base, and fuse all this information for text generation or loss calculation. The following components are essential to the model's functionality:

### **1. Image Encoder**
- The **Image Encoder** is based on the **Vision Transformer (ViT)**, which processes the input image and generates a feature embedding that represents the content of the image.
- The encoder captures spatial and semantic information from the image, which will later be fused with the query text and retrieved knowledge.

### **2. Text Encoder**
- The **Text Encoder** uses **T5**, a pre-trained transformer model, to process the input query text. It transforms the text into a feature vector that captures the semantic meaning of the query.

### **3. Retriever**
- The **Retriever** is responsible for fetching additional relevant information from the **knowledge base**. It concatenates the **encoded query text** and **image features** and projects them to the required dimension through an NN Layer to find the most relevant content stored in the knowledge base.
- The retriever uses **Faiss** for efficient similarity search of embeddings in the knowledge base. It retrieves the top-k most relevant items based on the query and image embeddings.

### **4. Fusion Layer**
- The **Fusion Layer** combines the encoded text, image features, retrieved knowledge embeddings and the projected vector required for the retrieval. This fusion process ensures that the model integrates information from all modalities (text, image, and knowledge) to create a comprehensive representation of the input data.
- The fusion is learnable with trainable parameters, which allow the model to adjust the relative importance of each modality (text, image, and retrieved knowledge).

### **5. Decoder**
- The **Decoder** is based on **T5** and is responsible for generating the final output. It processes the fused vector and outputs either a **textual description** (in inference mode) or **loss** (in training mode).
- The decoder is fine-tuned to handle the generated representations and produce coherent text outputs based on the multimodal input.


## 🏋️‍♂️ Training
The VLMRAG model is trained using a multi-task loss consisting of:

- Decoder Loss: Cross-entropy loss between generated text and ground truth captions.

- Retriever Loss: Cosine similarity between the retrieved knowledge embeddings and ground truth retrievals.

### Key Components:
- Optimizer: AdamW is used to optimize the model with a cosine annealing learning rate scheduler.

- Loss: Total loss = `decoder_loss + retr_lambda * retriever_loss`, where `retr_lambda` adjusts the weight of the retriever loss.

Training checkpoints are saved periodically, and training logs are written for later analysis.

Run the training script by executing the following command:
```
python train.py
```
The training parameters are stored in a configuration file (`config.yaml`)


## 📈 Evaluation

After training, evaluation is performed primarily by monitoring the **validation loss**, which combines decoder and retriever objectives.  

Validation logs are automatically saved during training in the experiment directory and can be inspected to understand model convergence and overfitting.

To visualize training and validation losses, as well as the evolution of the **learnable fusion parameters** (*alpha*, *beta*, *gamma*), the `analysis.ipynb` notebook can be referred to. 

Several challenges were encountered during training, and the steps taken to overcome them are detailed in the [challenges_and_solutions.md](docs/challenges_and_solutions.md) file.

## 🧠 Inference

Captions can be generated for custom images and queries using the `generate.py` script.

The script can be executed with the following command:

```
python generate.py
```

Ensure that the `ckpt_path` in the `config.yaml` is updated to point to the checkpoint that is to be used for inference.

The generated caption will be printed in the console. For visualization of the input image along with the output text, and for more detailed analysis of the generated captions, the `analysis.ipynb` notebook can be referred to.