import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image
import io

def visualize_heatmap(image_array, model, overlay_alpha=0.5):
    """
    Generate a heatmap visualization showing which parts of the image
    the model is focusing on for its prediction
    
    Args:
        image_array: preprocessed image numpy array
        model: the image model object
        overlay_alpha: transparency of the heatmap overlay
        
    Returns:
        heatmap_image: PIL Image with heatmap overlay
    """
    # Get the attention map from the model
    attention_map = model.get_attention_map(image_array)
    
    # Resize attention map to image size if needed
    if image_array.shape[0] == 1:  # Remove batch dimension if present
        img = image_array[0]
    else:
        img = image_array
        
    # Normalize image for display if needed
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    # Normalize attention map to [0, 1]
    attention_map = cv2.resize(attention_map, (img.shape[1], img.shape[0]))
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert from BGR to RGB if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1 - overlay_alpha, heatmap, overlay_alpha, 0)
    
    # Return as PIL Image for display in Streamlit
    return Image.fromarray(overlay)

def plot_prediction_scores(prediction, title="Prediction Scores"):
    """
    Create a bar chart showing prediction scores for different classes
    
    Args:
        prediction: numpy array of prediction scores
        title: chart title
        
    Returns:
        fig: matplotlib figure
    """
    classes = ["Pneumonia", "Tuberculosis", "COVID-19", "Healthy"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, prediction, color=['#FF9999', '#FFD700', '#FF6347', '#90EE90'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)  # Add some headroom for text
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance, title="Feature Importance"):
    """
    Create a horizontal bar chart showing the importance of clinical features
    
    Args:
        feature_importance: dictionary mapping feature names to importance scores
        title: chart title
        
    Returns:
        fig: matplotlib figure
    """
    # Sort by importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Human-readable feature names
    feature_display_names = {
        'age': 'Age',
        'temperature': 'Body Temperature',
        'heart_rate': 'Heart Rate',
        'oxygen_saturation': 'Oxygen Saturation',
        'wbc': 'White Blood Cell Count',
        'glucose': 'Blood Glucose',
        'creatinine': 'Creatinine Level',
        'systolic_bp': 'Systolic Blood Pressure',
        'diastolic_bp': 'Diastolic Blood Pressure',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'respiratory_disease': 'Respiratory Disease'
    }
    
    # Get display names
    names = [feature_display_names.get(f, f) for f in sorted_features.keys()]
    values = list(sorted_features.values())
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    bars = ax.barh(names, values, color='skyblue')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', ha='left', va='center')
    
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_plot(matrix, class_names=['Pneumonia', 'Tuberculosis', 'COVID-19', 'Healthy']):
    """
    Create a confusion matrix visualization
    
    Args:
        matrix: 2D numpy array of confusion matrix values
        class_names: list of class names
        
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with class names
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    
    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig

def visualize_learning_curves(history, title='Training History'):
    """
    Visualize learning curves from model training history
    
    Args:
        history: training history object from Keras
        title: plot title
        
    Returns:
        fig: matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1.plot(history.get('accuracy', []), label='Training Accuracy')
    ax1.plot(history.get('val_accuracy', []), label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    
    # Loss plot
    ax2.plot(history.get('loss', []), label='Training Loss')
    ax2.plot(history.get('val_loss', []), label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig