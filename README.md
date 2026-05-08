# Image-Processing
  Food Calories Estimation using RGB-D Fusion


# Features
- Augmentation – Enhances dataset diversity using random flips and shuffles to prevent overfitting and improve model generalization.

- Preprocessing – Cleans images using Median Blur for noise reduction and Kernel Sharpening to emphasize food textures.

- Edge Fusion – Extracts structural features using Sobel & Canny edge detection to help the model better understand food volume and height.

- Multi-modal Modeling – A hybrid architecture combining ResNet50 (for RGB features) and a Custom CNN (for Depth features).

- Training Optimization – Implements Learning Rate Reduction and Model Checkpointing to ensure the best weights are saved during training.

- Analytics & Prediction – Full evaluation using MAE and R2 Score, with an inference script for real-time calorie estimation on new images.
  
##  Project Structure
* **src/**: Core Python scripts (Data loading, Model, Training).
* **Plots/**: Visualization of MAE, Loss curves, and Prediction accuracy.
* **sample/**: Test images to try the model instantly.



##  Installation
```bash
git clone https://github.com/fouadmahmoud812-ai/Image-Processing.git
cd Image-Processing
pip install -r requirements.txt
```

##  Project Resources

*   ** Dataset (Kaggle):** [Nutrition5k Dataset](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset)
*   ** Interactive Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1VbsnNmCR2-UfS_r90KWGa3OQkhboR6ap?usp=sharing)


# Technologies Used

* Python 3.10+
* TensorFlow / Keras
* OpenCV
* NumPy & Pandas
* Matplotlib



