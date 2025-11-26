# Plant Disease Detection Web Application

## Live Web App

[Open Streamlit Plant Disease Detector](https://plant-disease-app-cbqdcjkgzfx2lj8drh855e.streamlit.app/)

This repository contains a deep learning based plant leaf disease classification system using EfficientNetB0 and deployed with Streamlit. The user can upload a leaf image and the model predicts the disease from 38 plant disease categories with a confidence score.

Model Notes & Usage Recommendations
-The model was trained on the PlantVillage dataset.
-For best results, upload images similar to dataset format:
 -clear leaves
 -close-up shots
 -minimal background
 -good lighting
 -Full-plant images or images with cluttered background may reduce accuracy.
Top-3 confidence scores are shown to help understand predictions.

---------------------------------------------------------------------

## Features

• Image-based plant disease recognition  
• Trained EfficientNetB0 model (.keras format)  
• Accepts JPG, JPEG, PNG images  
• Runs locally or on Streamlit Cloud  
• Provides prediction confidence percentage  
• No external dataset required for inference  

---------------------------------------------------------------------

## Repository Structure

plant-disease-app/
├── app.py                                  
├── plant_disease_effb0_best.keras          
├── class_names.json                         
├── plant-disease-detection.ipynb            
└── requirements.txt                        

---------------------------------------------------------------------

## Running Locally

1. Clone this repository

2. Install dependencies

3. Start the web app

Once executed, Streamlit will open the web interface automatically in a browser window.

---------------------------------------------------------------------

## Deploying to Streamlit Cloud

1. Upload the repository to GitHub  
2. Open https://share.streamlit.io  
3. Select the repository and choose `app.py` as the main file  
4. Ensure requirements.txt contains at minimum:
   streamlit
   tensorflow
   pillow
   numpy


5. Deploy and launch the web application.

---------------------------------------------------------------------

## How the Application Works

1.Loads the trained EfficientNetB0 .keras model using TensorFlow
2.User uploads a plant leaf image through the Streamlit interface
3.Image is resized to 224×224 and converted to a NumPy array (0–255 pixel values) — same preprocessing used during validation
4.Model predicts a probability distribution across 38 disease classes
5.The application displays:
  -Top predicted disease
  -Confidence score
(Also shows Top-3 probable classes)

---------------------------------------------------------------------

## Model Information

• Architecture: EfficientNetB0  
• Input Size: 224 × 224 × 3  
• Loss Function: Sparse Categorical Crossentropy  
• Achieved Validation Accuracy: ~99%  

---------------------------------------------------------------------

## Troubleshooting

| Problem                                | Solution                                                |
|----------------------------------------|---------------------------------------------------------|
| Model does not load                    | Verify `.keras` file exists in root folder             |
| Streamlit Cloud build fails            | Use `tensorflow-cpu` instead of `tensorflow`           |
| Predictions fail                       | Check versions of Pillow and NumPy                     |
| No output shown                        | Ensure image format is JPG/PNG and model loaded first  |

---------------------------------------------------------------------

## Possible Future Enhancements

• Grad-CAM heatmap visualization  
• Multi-image batch analysis  
• Camera-based real time predictions  
• Mobile app user interface  

---------------------------------------------------------------------


