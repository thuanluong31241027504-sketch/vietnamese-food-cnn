import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (
    Dense, Flatten, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, MaxPooling2D
)
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    def __init__(self, input_shape=(128, 128, 3), num_classes=30):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_cnn_model(self):
        """Xây dựng CNN model từ code gốc"""
        model = models.Sequential([
            Input(shape=self.input_shape),
            
            # Block 1
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(32, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.30),
            
            # Block 3
            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.35),
            
            # Block 4
            Conv2D(256, (3,3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.40),
            
            # Head
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model với optimizer và loss function"""
        if self.model is None:
            self.build_cnn_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def get_model_summary(self):
        """Lấy summary của model"""
        if self.model:
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            return "\n".join(string_list)
        return "Model chưa được khởi tạo"
