#  Butterfly Species Classifier

This project is a deep learning-based web application that classifies butterfly species using image input.

##  Features
- Upload butterfly image
- Predict top 3 species
- Confidence score display
- Species information (description, habitat, features)
- Sample image testing

##  Tech Stack
- TensorFlow (MobileNetV2)
- Streamlit
- Python

##  Model Accuracy
Achieved ~92–93% validation accuracy using transfer learning and fine-tuning.

##  Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
