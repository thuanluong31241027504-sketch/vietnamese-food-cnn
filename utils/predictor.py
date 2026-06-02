import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def preprocess_image(self, image_path, target_size=(128, 128)):
        """Tiền xử lý ảnh"""
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    
    def predict(self, image_path):
        """Dự đoán món ăn từ ảnh"""
        img_array, original_img = self.preprocess_image(image_path)
        
        # Dự đoán
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_food = self.class_names[predicted_class_idx]
        
        # Lấy top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (self.class_names[idx], predictions[0][idx])
            for idx in top_3_idx
        ]
        
        return {
            'predicted_food': predicted_food,
            'confidence': confidence,
            'top_3': top_3_predictions,
            'original_img': original_img,
            'all_probabilities': predictions[0]
        }
    
    def display_prediction(self, prediction_result):
        """Hiển thị kết quả dự đoán"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Hiển thị ảnh gốc
        ax1.imshow(prediction_result['original_img'])
        ax1.set_title(f"Dự đoán: {prediction_result['predicted_food']}\nĐộ tin cậy: {prediction_result['confidence']:.2%}")
        ax1.axis('off')
        
        # Hiển thị bar chart top predictions
        foods = [p[0] for p in prediction_result['top_3']]
        confidences = [p[1] for p in prediction_result['top_3']]
        
        colors = ['green' if i == 0 else 'blue' for i in range(len(foods))]
        ax2.barh(foods, confidences, color=colors)
        ax2.set_xlabel('Độ tin cậy')
        ax2.set_title('Top 3 dự đoán')
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        return fig
