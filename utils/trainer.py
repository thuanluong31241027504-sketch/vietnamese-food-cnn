import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, save_dir='models'):
        self.model = model
        self.save_dir = save_dir
        self.history = None
        os.makedirs(save_dir, exist_ok=True)
    
    def make_callbacks(self, save_name='best_model.keras', monitor='val_accuracy'):
        """Tạo callbacks cho training"""
        save_path = os.path.join(self.save_dir, save_name)
        
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_generator, valid_generator, epochs=30, save_name='best_model.keras'):
        """Train model"""
        callbacks = self.make_callbacks(save_name)
        
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Vẽ biểu đồ training history"""
        if self.history is None:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        return fig
    
    def save_model(self, filepath):
        """Lưu model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model
