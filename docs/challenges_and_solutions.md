# Challenges and Solutions

## 1. **Decoder Not Using Retrieved Vectors Effectively**
- **Challenge**: The decoder relies only on the text and image embeddings, with the retrieved vectors not being used. This leads to mildly relevant results. No gradient flow till the projection layer in the retriever module. Because, the projected tensor is only for faiss retrieval. Not being used for further calculations.
- **Solution**: 
    - Concatenate the projected tensor into the fused vector, which is then passed into the decoder, allowing gradient flow to the projection layer.
    - Use **cosine similarity** between the projected tensor and the Faiss retrievals, incorporating this as a penalty term to improve the quality of the retrieval.

## 2. **Retriever Loss Plateaus During Validation**
- **Challenge**: While both decoder and retriever losses decrease during training, the retriever loss plateaus after a certain point during validation, but the decoder loss shows minimal improvement, reducing slightly and then rising again in later epochs.
- **Cause**: The decoder relies heavily on the retrieved vectors, but the retrieval is so strong that the decoder stops improving on its own.
- **Solution**: 
    - Implemented **warm-up and cosine decay** for `retriever_lambda`, which helps regulate the retriever's influence over the training process and allows the decoder to keep improving.

## 3. **Poor Results on New Images**
- **Challenge**: The model performs poorly on new images, especially when trying to achieve broad descriptive output by only adjusting the projection layer before the decoder.
- **Solution**: **Unfreeze** the decoder parameters, at least for some layers, and train them as well to improve generalization and performance on new images.

## 4. **Fine-Tuning the Decoder and Encoder**
- **Challenge**: Fine-tuning up to some layers gave better results, as the model started to mimic the structure of the ground truth (GT) captions.
    - **Example**: 
        - **Image**: `data/original_val2017_img_and_captions/val2017/000000314709.jpg`
        - **Description**: "A boy jogs along the path of another male ox. A boy is a young male human, usually a child or an adolescent... (longer caption)"
- **Solution**: 
    1. Scale the **image embedding** contribution relative to the **retrieval embedding**.
    2. Sample **random retrievals** as distractors.
    3. Inspect the **retrieval content** to ensure quality.
    4. Keep the **encoders frozen** during fine-tuning.

## 5. **Improvements After Training More Decoder Layers**
- **Challenge**: After training one more decoder layer, results improved slightly. However, further fine-tuning is required.
    - **Dataset Size**: 5000 images.
    - **Solution**: 
        - Freeze only the **first decoder layer**.
        - Employ the following techniques to further improve results:
            1. Use **random distractor retrievals** to increase model robustness.
            2. Introduce **retrieval dropout** to force the model to focus on different parts of the retrieval.
            3. Use **scaled fusion** to fine-tune the contributions of image, text, and retrieval embeddings. Visualize in `analysis.ipynb`.
