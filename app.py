import sys
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra version Python
print(f"Python version: {sys.version}")

import streamlit as st
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

# Page config phải là lệnh đầu tiên sau khi import streamlit
st.set_page_config(
    page_title="Vietnamese Food CNN Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Hàm import tensorflow một cách an toàn
@st.cache_resource
def load_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        st.error(f"Không thể import TensorFlow: {e}")
        return None

tf = load_tensorflow()

# Kiểm tra nếu tensorflow không load được
if tf is None:
    st.error("""
    ❌ TensorFlow không thể load. Vui lòng kiểm tra lại cài đặt.
    
    Giải pháp: 
    1. Xóa cache của Streamlit Cloud
    2. Hoặc dùng phiên bản Python 3.9 trong runtime.txt
    3. Hoặc giảm bớt dependencies trong requirements.txt
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🍜 Vietnamese Food CNN")
    st.markdown("---")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    img_size = st.selectbox("Image Size", [(128, 128), (224, 224)], index=0)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32], value=16)  # Giảm batch size
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    epochs = st.slider("Epochs", min_value=1, max_value=10, value=5, step=1)  # Giảm epochs cho test
    
    st.markdown("---")
    st.info("💡 **Tip:** Training có thể mất vài phút với dataset nhỏ")

# Main content
st.markdown('<div class="main-header"><h1>🍜 Vietnamese Food Recognition with CNN</h1><p>Train CNN model trực tiếp và nhận diện món ăn Việt Nam</p></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Train Model", "🔮 Predict"])

# Tab 1: Train
with tab1:
    st.subheader("🎯 Train CNN Model")
    
    # Chỉ cho phép train với dataset nhỏ
    st.warning("⚠️ **Lưu ý:** Do giới hạn RAM của Streamlit Cloud, chỉ nên train với số lượng ảnh nhỏ (100-200 ảnh/class)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Chuẩn bị dữ liệu")
        
        # Option 1: Upload data
        use_sample = st.checkbox("Sử dụng sample data (khuyến nghị)")
        
        if use_sample:
            st.info("Sẽ tạo sample data với 5 class để test")
            if st.button("📁 Tạo Sample Data", type="primary"):
                with st.spinner("Đang tạo sample data..."):
                    # Tạo sample data trong memory
                    import tempfile
                    import zipfile
                    from io import BytesIO
                    from PIL import Image, ImageDraw
                    
                    # Tạo thư mục tạm
                    temp_dir = tempfile.mkdtemp()
                    
                    # Tạo 5 class giả
                    classes = ['Pho', 'Bun Cha', 'Banh Mi', 'Com Tam', 'Banh Xeo']
                    st.session_state.class_names = classes
                    
                    # Tạo ảnh giả
                    for class_name in classes:
                        class_dir = os.path.join(temp_dir, 'Train', class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        # Tạo 10 ảnh giả
                        for i in range(10):
                            img = Image.new('RGB', (128, 128), color=(np.random.randint(0,255), 
                                                                     np.random.randint(0,255), 
                                                                     np.random.randint(0,255)))
                            img.save(os.path.join(class_dir, f'{class_name}_{i}.jpg'))
                    
                    st.session_state.data_path = temp_dir
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data đã được tạo với 5 classes!")
        else:
            st.info("Upload dataset của bạn (cấu trúc: Train/Class_name/images.jpg)")
            uploaded_zip = st.file_uploader("Upload file ZIP", type=['zip'])
            if uploaded_zip and st.button("Extract"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "data.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.getbuffer())
                    
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    
                    st.session_state.data_path = tmpdir
                    st.session_state.data_loaded = True
                    st.success("✅ Dữ liệu đã được giải nén!")
    
    with col2:
        st.markdown("### 2. Xây dựng và Train model")
        
        if st.session_state.data_loaded:
            if st.button("🏗️ Build & Train Model", type="primary"):
                with st.spinner("Đang xây dựng model..."):
                    try:
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense, Input
                        from tensorflow.keras.optimizers import Adam
                        from tensorflow.keras.preprocessing.image import ImageDataGenerator
                        
                        # Build model đơn giản hơn để tiết kiệm RAM
                        model = Sequential([
                            Input(shape=(img_size[0], img_size[1], 3)),
                            Conv2D(16, (3,3), activation='relu'),
                            MaxPooling2D(),
                            Conv2D(32, (3,3), activation='relu'),
                            MaxPooling2D(),
                            Conv2D(64, (3,3), activation='relu'),
                            MaxPooling2D(),
                            Flatten(),
                            Dense(64, activation='relu'),
                            Dropout(0.5),
                            Dense(len(st.session_state.class_names) if st.session_state.class_names else 5, activation='softmax')
                        ])
                        
                        model.compile(optimizer=Adam(learning_rate=learning_rate),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
                        
                        st.success("✅ Model đã được xây dựng!")
                        
                        # Load data
                        st.info("Đang load dữ liệu...")
                        train_datagen = ImageDataGenerator(rescale=1./255)
                        
                        train_path = os.path.join(st.session_state.data_path, 'Train')
                        train_generator = train_datagen.flow_from_directory(
                            train_path,
                            target_size=img_size,
                            batch_size=batch_size,
                            class_mode='categorical'
                        )
                        
                        # Training
                        st.info(f"🚀 Bắt đầu training với {epochs} epochs...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Custom callback để update progress
                        class ProgressCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Epoch {epoch+1}/{epochs} - Accuracy: {logs.get('accuracy', 0):.3f}")
                        
                        history = model.fit(
                            train_generator,
                            epochs=epochs,
                            callbacks=[ProgressCallback()],
                            verbose=0
                        )
                        
                        st.session_state.model = model
                        st.session_state.model_trained = True
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training hoàn tất!")
                        st.success("✅ Model đã được training thành công!")
                        
                        # Plot results
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(history.history['accuracy'], label='Training Accuracy')
                        ax.set_title('Model Accuracy')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")
                        st.info("💡 Hãy thử giảm batch_size hoặc epochs")
        else:
            st.info("👉 Hãy chuẩn bị dữ liệu ở cột bên trái trước")

# Tab 2: Predict
with tab2:
    st.subheader("🔍 Nhận diện món ăn")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Vui lòng train model trước khi dự đoán!")
        
        # Option to upload pre-trained model
        st.subheader("Hoặc upload model đã train sẵn:")
        uploaded_model = st.file_uploader("Upload model file (.keras)", type=['keras'])
        if uploaded_model and st.button("Load Model"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
            
            st.session_state.model = tf.keras.models.load_model(model_path)
            if st.session_state.class_names is None:
                st.session_state.class_names = ['Pho', 'Bun Cha', 'Banh Mi', 'Com Tam', 'Banh Xeo']
            st.session_state.model_trained = True
            st.success("✅ Model loaded!")
    else:
        uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh của bạn", use_column_width=True)
            
            if st.button("🔍 Dự đoán", type="primary"):
                with st.spinner("Đang xử lý..."):
                    # Preprocess
                    img = image.resize(img_size)
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    predictions = st.session_state.model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    # Display result
                    st.success(f"### 🎯 Kết quả: **{st.session_state.class_names[predicted_class]}**")
                    st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                    
                    # Show top 3
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    st.subheader("🏆 Top 3 dự đoán:")
                    for idx in top_3_idx:
                        st.progress(predictions[0][idx], text=f"{st.session_state.class_names[idx]}: {predictions[0][idx]:.2%}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Built with ❤️ using TensorFlow & Streamlit</div>", unsafe_allow_html=True)
