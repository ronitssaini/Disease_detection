import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Attention

class FusionModel:
    """
    Model for combining predictions from multiple modalities (image and clinical data)
    """
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build and compile the fusion model
        
        This model takes predictions from individual models and combines them
        using a sophisticated attention mechanism.
        """
        # Define inputs: predictions from each modality
        image_input = Input(shape=(self.num_classes,), name='image_prediction')
        clinical_input = Input(shape=(self.num_classes,), name='clinical_prediction')
        
        # Use attention mechanism to weigh the predictions
        attention_layer = Attention()([image_input, clinical_input])
        
        # Concatenate the attention output with original inputs
        concat = Concatenate()([image_input, clinical_input, attention_layer])
        
        # Final dense layer for integration
        output = Dense(self.num_classes, activation='softmax', name='fusion_output')(concat)
        
        # Create the model
        model = Model(
            inputs=[image_input, clinical_input],
            outputs=output
        )
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def combine_predictions(self, image_prediction, clinical_prediction):
        """
        Combine predictions from image and clinical models
        
        In a real application, this would use the actual trained model.
        For demonstration, we're using a simulated approach.
        """
        # Ensure predictions are numpy arrays
        image_pred = np.array(image_prediction)
        clinical_pred = np.array(clinical_prediction)
        
        # This is a placeholder for demonstration
        # In a real application, you would use:
        # return self.model.predict([image_pred[np.newaxis, ...], clinical_pred[np.newaxis, ...]])[0]
        
        # For demonstration, use a weighted average with confidence-based weights
        image_confidence = np.max(image_pred)
        clinical_confidence = np.max(clinical_pred)
        
        # Calculate weights based on confidence
        total_confidence = image_confidence + clinical_confidence
        image_weight = image_confidence / total_confidence
        clinical_weight = clinical_confidence / total_confidence
        
        # Apply additive attention mechanism (simplified for demonstration)
        # In real implementation, use a learned attention mechanism
        combined_pred = (image_pred * image_weight) + (clinical_pred * clinical_weight)
        
        # Ensure the result is a valid probability distribution
        return combined_pred / np.sum(combined_pred)
    
    def get_modality_weights(self, image_prediction, clinical_prediction):
        """
        Return the importance weight assigned to each modality
        
        For explainability purposes
        """
        image_pred = np.array(image_prediction)
        clinical_pred = np.array(clinical_prediction)
        
        image_confidence = np.max(image_pred)
        clinical_confidence = np.max(clinical_pred)
        
        total_confidence = image_confidence + clinical_confidence
        image_weight = image_confidence / total_confidence
        clinical_weight = clinical_confidence / total_confidence
        
        return {
            'image_weight': image_weight,
            'clinical_weight': clinical_weight
        }
    
    def load_weights(self, weights_path):
        """Load trained weights for the model"""
        try:
            self.model.load_weights(weights_path)
            return True
        except:
            print(f"Error loading weights from {weights_path}")
            return False