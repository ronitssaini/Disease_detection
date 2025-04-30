import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import cv2

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for the image model
    
    Args:
        image: PIL Image object
        target_size: tuple of (height, width) for resizing
        
    Returns:
        preprocessed_image: numpy array ready for model input
    """
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        # Resize the image
        image = image.resize(target_size)
        # Convert to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        img_array = np.array(image)
    else:
        # If already numpy array, resize it
        img_array = cv2.resize(np.array(image), target_size)
        # Convert BGR to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension if not present
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_clinical_data(data):
    """
    Preprocess clinical data for the clinical model
    
    Args:
        data: dictionary of clinical features
        
    Returns:
        preprocessed_data: processed data ready for model input
    """
    # Create a copy to avoid modifying the original
    processed = data.copy()
    
    # One-hot encode categorical variables
    if 'gender' in processed:
        gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
        processed['gender_encoded'] = gender_mapping.get(processed['gender'], 0)
        del processed['gender']
    
    # Normalize continuous variables to [0,1] range
    # Age normalization (assuming max age of 100)
    if 'age' in processed:
        processed['age'] = processed['age'] / 100.0
    
    # Temperature normalization (assuming range 35-42°C)
    if 'temperature' in processed:
        processed['temperature'] = (processed['temperature'] - 35) / 7.0
        
    # Heart rate normalization (assuming range 40-200 bpm)
    if 'heart_rate' in processed:
        processed['heart_rate'] = (processed['heart_rate'] - 40) / 160.0
        
    # Blood pressure normalization
    if 'systolic_bp' in processed:
        processed['systolic_bp'] = (processed['systolic_bp'] - 70) / 180.0
    
    if 'diastolic_bp' in processed:
        processed['diastolic_bp'] = (processed['diastolic_bp'] - 40) / 110.0
    
    # Oxygen saturation normalization (already in percent, but normalize to [0,1])
    if 'oxygen_saturation' in processed:
        processed['oxygen_saturation'] = (processed['oxygen_saturation'] - 70) / 30.0
    
    # Lab values normalization
    if 'wbc' in processed:
        processed['wbc'] = processed['wbc'] / 30.0  # Normalize WBC count
    
    if 'hgb' in processed:
        processed['hgb'] = processed['hgb'] / 20.0  # Normalize hemoglobin
    
    if 'glucose' in processed:
        processed['glucose'] = processed['glucose'] / 500.0  # Normalize glucose
    
    if 'creatinine' in processed:
        processed['creatinine'] = processed['creatinine'] / 10.0  # Normalize creatinine
    
    # Convert boolean values to integers
    bool_features = ['diabetes', 'hypertension', 'heart_disease', 'respiratory_disease']
    for feature in bool_features:
        if feature in processed:
            processed[feature] = 1 if processed[feature] else 0
    
    # In a real application, you might handle missing values, 
    # feature engineering, etc. here
    
    return processed

def create_feature_vector(processed_clinical_data):
    """
    Convert preprocessed clinical data dictionary to a feature vector
    that can be used by the model
    
    Args:
        processed_clinical_data: preprocessed dictionary
        
    Returns:
        feature_vector: numpy array
    """
    # Define the expected features and their order
    expected_features = [
        'age', 'gender_encoded', 'temperature', 'heart_rate', 
        'systolic_bp', 'diastolic_bp', 'oxygen_saturation',
        'wbc', 'hgb', 'glucose', 'creatinine',
        'diabetes', 'hypertension', 'heart_disease', 'respiratory_disease'
    ]
    
    # Create the feature vector
    feature_vector = np.zeros(len(expected_features))
    
    # Fill in available values
    for i, feature in enumerate(expected_features):
        if feature in processed_clinical_data:
            feature_vector[i] = processed_clinical_data[feature]
    
    # Add batch dimension
    feature_vector = np.expand_dims(feature_vector, axis=0)
    
    return feature_vector

def load_sample_data():
    """
    Load sample data for testing/demonstration
    
    Returns:
        sample_image: a sample medical image
        sample_clinical_data: a sample clinical data dict
    """
    # This is a placeholder - in a real app, load actual sample data
    # For demonstration, we'll create synthetic data
    
    # Create a synthetic sample image (gray square with a circle)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Gray background
    cv2.circle(img, (112, 112), 50, (100, 100, 100), -1)  # Dark circle
    sample_image = Image.fromarray(img)
    
    # Create sample clinical data
    sample_clinical_data = {
        'age': 65,
        'gender': 'Male',
        'temperature': 38.2,
        'heart_rate': 95,
        'systolic_bp': 130,
        'diastolic_bp': 85,
        'oxygen_saturation': 94,
        'wbc': 12.5,
        'hgb': 13.5,
        'glucose': 110,
        'creatinine': 1.2,
        'diabetes': True,
        'hypertension': True,
        'heart_disease': False,
        'respiratory_disease': True
    }
    
    return sample_image, sample_clinical_data