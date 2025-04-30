import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

class ImageModel:
    """
    Model for processing medical images and making predictions
    """
    def __init__(self, num_classes=4, img_size=(224, 224)):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and compile the model architecture"""
        # Use a pre-trained model as the base
        base_model = DenseNet121(
            weights='imagenet',  # In a real app, you might use specific medical pre-trained weights
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # In a real application, only fine-tune the top layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def predict(self, preprocessed_image):
        """
        Make a prediction on a preprocessed image
        
        In a real application, this would use the actual trained model.
        For demonstration, we're returning simulated predictions.
        """
        # This is a placeholder for demonstration
        # In a real application, you would use:
        # return self.model.predict(preprocessed_image)
        
        # Simulated prediction (probability distribution across classes)
        # Classes: ["Pneumonia", "Tuberculosis", "COVID-19", "Healthy"]
        if np.random.random() > 0.5:
            # Simulate a case with disease
            return np.array([0.75, 0.15, 0.05, 0.05])
        else:
            # Simulate a healthy case
            return np.array([0.05, 0.05, 0.05, 0.85])
    
    def get_attention_map(self, preprocessed_image):
        """
        Generate an attention map highlighting regions the model is focusing on
        
        For demonstration, this returns a simulated heatmap.
        In a real implementation, this would use Grad-CAM or similar techniques.
        """
        # This is a placeholder - in a real app, implement Grad-CAM or similar
        # Simulate an attention heatmap
        return np.random.rand(*preprocessed_image.shape[:2])
    
    def load_weights(self, weights_path):
        """Load trained weights for the model"""
        try:
            self.model.load_weights(weights_path)
            return True
        except:
            print(f"Error loading weights from {weights_path}")
            return False