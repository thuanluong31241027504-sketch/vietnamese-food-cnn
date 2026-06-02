import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import shutil
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.class_names = None
        
    def download_from_kaggle(self):
        """Download dataset from Kaggle"""
        import kagglehub
        with st.spinner("Đang tải dataset từ Kaggle..."):
            path = kagglehub.dataset_download("quandang/vietnamese-foods")
            return path
    
    def create_dataframe(self, directory):
        """Tạo dataframe từ thư mục ảnh"""
        filepaths, labels = [], []
        if not os.path.exists(directory):
            return pd.DataFrame({'filepath': [], 'label': []})
        
        for label in os.listdir(directory):
            class_dir = os.path.join(directory, label)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        filepaths.append(os.path.join(class_dir, file))
                        labels.append(label)
        return pd.DataFrame({'filepath': filepaths, 'label': labels})
    
    def load_data(self, base_dir, img_size=(128, 128), batch_size=32):
        """Load và prepare dữ liệu"""
        train_df = self.create_dataframe(os.path.join(base_dir, 'Train'))
        valid_df = self.create_dataframe(os.path.join(base_dir, 'Validate'))
        test_df = self.create_dataframe(os.path.join(base_dir, 'Test'))
        
        self.class_names = sorted(train_df['label'].unique())
        
        # Data augmentation cho training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Chỉ rescale cho validation và test
        valid_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='label',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        valid_generator = valid_test_datagen.flow_from_dataframe(
            valid_df,
            x_col='filepath',
            y_col='label',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        test_generator = valid_test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepath',
            y_col='label',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, valid_generator, test_generator
