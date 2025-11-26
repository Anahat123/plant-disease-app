# Plant Disease Detection Web Application

This repository contains a deep learning based plant leaf disease classification system using EfficientNetB0 and deployed with Streamlit. The user can upload a leaf image and the model predicts the disease from 38 plant disease categories with a confidence score.

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

1. Loads EfficientNetB0 `.keras` model using TensorFlow  
2. User uploads a plant leaf image  
3. Image is resized to 224x224 and converted into a normalized NumPy array  
4. Model predicts the probability distribution over 38 classes  
5. Application displays:
   - Predicted disease name  
   - Confidence score  

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


