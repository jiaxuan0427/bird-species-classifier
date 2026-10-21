# Bird Species Classification Using EfficientNet-B0 and PyTorch
An AI-powered web application that classifies images into 200 bird species using a fine-tuned EfficientNet-B0 model, PyTorch, and Streamlit. ResNet18 was also explored during model experimentation.

## 1. Access the Web Application
**Public URL:** https://bird-species-classifier-jx.streamlit.app/ 

Open the link in a web browser, upload a bird image (JPG, JPEG, or PNG), and click **Identify This Bird** to view the predicted species and Top-5 predictions.

**No installation is required** to use the public website.

## 2. Run Locally
### Requirements
- Python 3.10 or a compatible Python version
- Project source code
- Trained model checkpoint: `outputs/efficientnet_b0_best.pth`

### Install dependencies
Open a terminal in the project directory and run:
```bash
python -m pip install streamlit torch torchvision pillow numpy scikit-learn
```

### Start the application
```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 3. Model Evaluation
To evaluate the model on the official test dataset, ensure the test images and `data/test.txt` are available, then run:
```bash
python evaluate_efficientnet.py
```

The evaluation script reports Top-1 accuracy, Top-5 accuracy, average class accuracy, macro precision, macro recall, and macro F1-score. It also saves the confusion matrix.

## 4. Important Files
- `app.py` — Streamlit web application.
- `outputs/efficientnet_b0_best.pth` — Trained EfficientNet-B0 checkpoint.
- `evaluate_efficientnet.py` — Model evaluation script.
- `data/Test/` and `data/test.txt` — Official test images and annotations required for evaluation.

## 5. Notes
- The trained checkpoint must be available at the specified path to run the application.
- The test dataset is required only for running the evaluation script.
- Predictions are estimates and may not always be correct.

**Final model:** EfficientNet-B0  
**Framework:** PyTorch  
**Web interface:** Streamlit