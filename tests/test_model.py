import unittest
import tensorflow as tf
import numpy as np
from src.models.cnn_model import create_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = create_model()
    
    def test_model_output_shape(self):
        # Create a dummy input
        dummy_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
        
        # Get the model output
        output = self.model(dummy_input)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 10))
    
    def test_model_training(self):
        # Create dummy data
        train_images = np.random.rand(100, 32, 32, 3).astype(np.float32)
        train_labels = np.random.randint(0, 10, size=(100,))
        
        # Train the model
        history = self.model.fit(train_images, train_labels, epochs=1)
        
        # Check if the model has been trained
        self.assertGreater(history.history['accuracy'][0], 0)
    
    def test_model_compilation(self):
        # Check if the model is compiled with the right loss and optimizer
        self.assertIsInstance(self.model.loss, tf.keras.losses.SparseCategoricalCrossentropy)
        self.assertIsInstance(self.model.optimizer, tf.keras.optimizers.Adam)
    
if __name__ == '__main__':
    unittest.main()
