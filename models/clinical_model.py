import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization

class ClinicalModel:
    """
    Model for processing clinical data and making predictions
    """
    def __init__(self, num_classes=4, input_dim=20):
        self.input_dim = input_dim  # Number of clinical features after preprocessing
        self.num_classes = num_classes
        self.model = self._build_model()
        self.feature_importance = None
        
    def _build_model(self):
        """Build and compile the model architecture"""
        # Define the model architecture for clinical data
        inputs = Input(shape=(self.input_dim,))
        
        # First hidden layer
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second hidden layer
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create and compile the model
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, preprocessed_clinical_data):
        """
        Make a prediction based on clinical data
        
        In a real application, this would use the actual trained model.
        For demonstration, we're returning simulated predictions.
        """
        # This is a placeholder for demonstration
        # In a real application, you would use:
        # return self.model.predict(preprocessed_clinical_data)
        
        # Simulated prediction (probability distribution across classes)
        # Classes: ["Pneumonia", "Tuberculosis", "COVID-19", "Healthy"]
        
        # Extract some key features to drive our simulation
        has_fever = False
        has_respiratory_condition = False
        
        if isinstance(preprocessed_clinical_data, dict):
            has_fever = preprocessed_clinical_data.get('temperature', 37.0) > 38.0
            has_respiratory_condition = preprocessed_clinical_data.get('respiratory_disease', False)
            oxygen_level = preprocessed_clinical_data.get('oxygen_saturation', 98)
            
            # Calculate feature importance for visualization
            self.feature_importance = self._calculate_feature_importance(preprocessed_clinical_data)
        else:
            # If preprocessed data is already a numpy array, use random values
            has_fever = np.random.random() > 0.7
            has_respiratory_condition = np.random.random() > 0.7
            oxygen_level = np.random.randint(92, 100)
        
        # Simulate predictions based on these features
        if has_fever and has_respiratory_condition and oxygen_level < 95:
            # High chance of pneumonia or COVID
            if np.random.random() > 0.5:
                return np.array([0.7, 0.1, 0.15, 0.05])  # Pneumonia
            else:
                return np.array([0.15, 0.05, 0.75, 0.05])  # COVID
        elif has_fever and has_respiratory_condition:
            # Moderate chance of respiratory disease
            return np.array([0.4, 0.3, 0.2, 0.1])
        elif has_fever:
            # Some illness but maybe not respiratory
            return np.array([0.3, 0.2, 0.1, 0.4])
        else:
            # Likely healthy
            return np.array([0.05, 0.05, 0.1, 0.8])
    
    def _calculate_feature_importance(self, data):
        """
        Calculate the importance of each clinical feature
        This would typically use techniques like SHAP values
        For demonstration, we're simulating the importance
        """
        # Define the features we want to show importance for
        features = [
            'age', 'temperature', 'heart_rate', 'oxygen_saturation', 
            'wbc', 'glucose', 'creatinine', 'systolic_bp'
        ]
        
        # Create simulated importance scores
        importance = {}
        
        for feature in features:
            if feature in data:
                # Assign higher importance to abnormal values
                if feature == 'temperature' and data[feature] > 38.0:
                    importance[feature] = 0.8
                elif feature == 'oxygen_saturation' and data[feature] < 95:
                    importance[feature] = 0.9
                elif feature == 'wbc' and (data[feature] < 4.0 or data[feature] > 11.0):
                    importance[feature] = 0.7
                else:
                    importance[feature] = np.random.random() * 0.5
            else:
                importance[feature] = np.random.random() * 0.3
                
        return importance
    
    def get_feature_importance(self):
        """Return the calculated feature importance"""
        if self.feature_importance is None:
            # Return some defaults if no prediction has been made yet
            features = [
                'age', 'temperature', 'heart_rate', 'oxygen_saturation', 
                'wbc', 'glucose', 'creatinine', 'systolic_bp'
            ]
            return {feature: np.random.random() for feature in features}
        
        return self.feature_importance
    
    def load_weights(self, weights_path):
        """Load trained weights for the model"""
        try:
            self.model.load_weights(weights_path)
            return True
        except:
            print(f"Error loading weights from {weights_path}")
            return False