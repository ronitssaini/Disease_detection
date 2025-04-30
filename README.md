# Multi-Modal Disease Detection System

## Project Overview

This project implements a multi-modal disease detection system that combines medical imaging and clinical data for more accurate disease classification. The system uses deep learning techniques and a novel fusion approach to integrate different data modalities.

**Project by Roll No. 81-83**

### Features

- **Medical Image Analysis**: Analyze X-rays and other medical images using deep learning
- **Clinical Data Processing**: Process patient clinical data including demographics, vital signs, and lab values
- **Multi-modal Fusion**: Novel fusion approach to combine predictions from different data sources
- **Interactive UI**: User-friendly Streamlit interface for easy interaction with the system
- **Visualization**: Attention maps and feature importance visualizations for interpretability

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/multimodal-disease-detection.git
   cd multimodal-disease-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit app:
```
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
multimodal_disease_detection/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── models/
│   ├── image_model.py          # Image processing model
│   ├── clinical_model.py       # Clinical data processing model
│   └── fusion_model.py         # Model fusion implementation
├── utils/
│   ├── preprocess.py           # Data preprocessing functions
│   └── visualization.py        # Visualization utilities
└── data/
    ├── sample_images/          # Sample medical images for testing
    └── sample_clinical_data.csv # Sample clinical data for testing
```

## Usage

1. **Upload Medical Image**: Upload an X-ray or other medical image for analysis, or use the provided samples.
2. **Enter Clinical Data**: Input patient information, vital signs, and lab values.
3. **Analyze**: Click the analyze button to process the data and view the results.
4. **Interpret Results**: View the attention heatmap, clinical feature importance, and final disease prediction.

## Technical Details

### Models

- **Image Model**: DenseNet121-based convolutional neural network fine-tuned for medical image analysis
- **Clinical Model**: Multi-layer neural network for processing clinical parameters
- **Fusion Model**: Attention-based fusion mechanism to combine predictions from both modalities

### Data Processing

- Images are preprocessed (resized, normalized) before being fed to the model
- Clinical data is normalized and encoded appropriately
- Feature importance is calculated to provide interpretability

## Future Improvements

- Add support for more imaging modalities (CT, MRI, etc.)
- Incorporate natural language processing for medical text records
- Implement more advanced fusion mechanisms
- Add support for more disease categories
- Improve model explainability and visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical imaging datasets used for model training
- TensorFlow and Keras communities
- Streamlit for the excellent web app framework