import unittest
import numpy as np
from src.utils.data_preprocessing import normalize_data

class TestDataPreprocessing(unittest.TestCase):
    def test_normalize_data(self):
        # Create a dummy dataset
        train_images = np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8)
        test_images = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
        
        # Normalize the data
        train_images_norm, test_images_norm = normalize_data(train_images, test_images)
        
        # Check the range of the normalized data
        self.assertTrue(np.all(train_images_norm >= 0) and np.all(train_images_norm <= 1))
        self.assertTrue(np.all(test_images_norm >= 0) and np.all(test_images_norm <= 1))
        
        # Check the shape of the normalized data
        self.assertEqual(train_images_norm.shape, train_images.shape)
        self.assertEqual(test_images_norm.shape, test_images.shape)

if __name__ == '__main__':
    unittest.main()
