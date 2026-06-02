import streamlit as st
import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time

# Import utils
from utils.data_loader import DataLoader
from utils.model_builder import ModelBuilder
from utils.trainer import Trainer
from utils.predictor import Predictor

# Page config
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
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1999/1999625.png", width=100)
    st.title("🍜 Vietnamese Food CNN")
    st.markdown("---")
    
    # Parameters
    st.subheader("⚙️ Parameters")
    img_size = st.selectbox("Image Size", [(128, 128), (224, 224)], index=0)
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    epochs = st.slider("Epochs", min_value=5, max_value=100, value=30, step=5)
    
    st.markdown("---")
    st.info("💡 **Tip:** Training có thể mất vài phút. Hãy kiên nhẫn!")

# Main content
st.markdown('<div class="main-header"><h1>🍜 Vietnamese Food Recognition with CNN</h1><p>Train CNN model trực tiếp và nhận diện món ăn Việt Nam</p></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Load & Train", "🔮 Predict", "📈 Training History"])

# Tab 1: Load and Train
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Load Dataset")
        
        # Option to download or upload
        data_option = st.radio(
            "Chọn cách lấy dữ liệu:",
            ["📥 Tự động download từ Kaggle", "📁 Upload thư mục dữ liệu"]
        )
        
        if data_option == "📥 Tự động download từ Kaggle":
            if st.button("🔄 Download Dataset từ Kaggle", type="primary"):
                with st.spinner("Đang download dataset từ Kaggle..."):
                    try:
                        dataloader = DataLoader()
                        path = dataloader.download_from_kaggle()
                        st.session_state.data_path = path
                        st.session_state.data_loaded = True
                        st.success(f"✅ Dataset đã được tải về: {path}")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tải dataset: {str(e)}")
        else:
            uploaded_dir = st.file_uploader(
                "Upload thư mục dữ liệu (file zip)",
                type=['zip']
            )
            if uploaded_dir and st.button("Extract Data"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "data.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_dir.getbuffer())
                    
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                    st.session_state.data_path = tmpdir
                    st.session_state.data_loaded = True
                    st.success("✅ Dữ liệu đã được giải nén!")
        
        # Load data button
        if st.session_state.data_loaded and st.button("📂 Load Data vào Generator", type="primary"):
            with st.spinner("Đang load dữ liệu..."):
                try:
                    dataloader = DataLoader()
                    train_gen, valid_gen, test_gen = dataloader.load_data(
                        st.session_state.data_path,
                        img_size=img_size,
                        batch_size=batch_size
                    )
                    
                    st.session_state.train_generator = train_gen
                    st.session_state.valid_generator = valid_gen
                    st.session_state.test_generator = test_gen
                    st.session_state.class_names = dataloader.class_names
                    
                    st.success(f"✅ Load thành công! Found {len(train_gen.classes)} classes")
                    st.info(f"Classes: {', '.join(st.session_state.class_names[:5])}...")
                except Exception as e:
                    st.error(f"❌ Lỗi khi load data: {str(e)}")
    
    with col2:
        st.subheader("2️⃣ Build & Train Model")
        
        if st.button("🏗️ Build CNN Model", type="primary"):
            with st.spinner("Đang xây dựng CNN model..."):
                try:
                    builder = ModelBuilder(
                        input_shape=(img_size[0], img_size[1], 3),
                        num_classes=len(st.session_state.class_names) if st.session_state.class_names else 30
                    )
                    model = builder.build_cnn_model()
                    model = builder.compile_model(learning_rate=learning_rate)
                    
                    st.session_state.model = model
                    st.session_state.builder = builder
                    
                    st.success("✅ Model CNN đã được xây dựng!")
                    
                    # Show model summary
                    with st.expander("📋 Model Summary"):
                        summary_text = builder.get_model_summary()
                        st.code(summary_text, language="python")
                except Exception as e:
                    st.error(f"❌ Lỗi khi build model: {str(e)}")
        
        if st.session_state.model is not None and st.session_state.get('train_generator') is not None:
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Đang training model..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        trainer = Trainer(st.session_state.model, save_dir='models')
                        
                        # Create callback for progress update
                        class ProgressCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress_bar.progress((epoch + 1) / epochs)
                                status_text.text(f"Epoch {epoch+1}/{epochs} - Accuracy: {logs.get('accuracy', 0):.4f} - Val Accuracy: {logs.get('val_accuracy', 0):.4f}")
                        
                        history = trainer.train(
                            st.session_state.train_generator,
                            st.session_state.valid_generator,
                            epochs=epochs,
                            save_name='vietnamese_food_cnn.keras'
                        )
                        
                        st.session_state.trainer = trainer
                        st.session_state.history = history
                        st.session_state.model_trained = True
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training hoàn tất!")
                        st.success("✅ Model đã được training và lưu thành công!")
                        
                        # Plot immediate results
                        fig = trainer.plot_training_history()
                        if fig:
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"❌ Lỗi khi training: {str(e)}")

# Tab 2: Predict
with tab2:
    st.subheader("🔍 Nhận diện món ăn Việt Nam")
    
    if not st.session_state.model_trained or st.session_state.model is None:
        st.warning("⚠️ Vui lòng load data và train model trước khi dự đoán!")
        
        # Option to load pre-trained model
        st.subheader("Hoặc load model đã train sẵn:")
        uploaded_model = st.file_uploader("Upload model file (.keras)", type=['keras'])
        if uploaded_model and st.button("Load Model"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
            
            st.session_state.model = tf.keras.models.load_model(model_path)
            st.session_state.model_trained = True
            st.success("✅ Model đã được load thành công!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Upload image
            uploaded_file = st.file_uploader(
                "Chọn ảnh món ăn",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload ảnh món ăn Việt Nam để nhận diện"
            )
            
            if uploaded_file:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Ảnh đã upload", use_column_width=True)
                
                if st.button("🔍 Predict", type="primary"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Predict
                    predictor = Predictor(st.session_state.model, st.session_state.class_names)
                    result = predictor.predict(temp_path)
                    
                    st.session_state.prediction_result = result
        
        with col2:
            if 'prediction_result' in st.session_state:
                result = st.session_state.prediction_result
                
                st.success(f"### 🎯 Dự đoán: **{result['predicted_food']}**")
                st.info(f"📊 Độ tin cậy: **{result['confidence']:.2%}**")
                
                # Display top 3 predictions
                st.subheader("🏆 Top 3 dự đoán:")
                for i, (food, conf) in enumerate(result['top_3'], 1):
                    st.progress(conf, text=f"{i}. {food}: {conf:.2%}")
                
                # Plot prediction chart
                fig = predictor.display_prediction(result)
                st.pyplot(fig)
                
                # Clean up
                os.unlink(temp_path)

# Tab 3: Training History
with tab3:
    if st.session_state.get('trainer') and st.session_state.trainer.history:
        st.subheader("📊 Training Results")
        
        # Display metrics
        history = st.session_state.trainer.history
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Train Accuracy", f"{history.history['accuracy'][-1]:.2%}")
        with col2:
            st.metric("Final Val Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
        with col3:
            st.metric("Final Train Loss", f"{history.history['loss'][-1]:.4f}")
        with col4:
            st.metric("Final Val Loss", f"{history.history['val_loss'][-1]:.4f}")
        
        # Plot training history
        st.subheader("📈 Training Curves")
        fig = st.session_state.trainer.plot_training_history()
        if fig:
            st.pyplot(fig)
        
        # Download model button
        st.subheader("💾 Download Model")
        if st.button("Download Trained Model"):
            model_path = 'models/vietnamese_food_cnn.keras'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="📥 Click để download",
                        data=f,
                        file_name="vietnamese_food_cnn.keras",
                        mime="application/octet-stream"
                    )
            else:
                st.error("Không tìm thấy file model!")
        
        # Display training table
        with st.expander("📋 Chi tiết từng epoch"):
            df_history = pd.DataFrame(history.history)
            st.dataframe(df_history, use_container_width=True)
    else:
        st.info("💡 Chưa có dữ liệu training nào. Hãy train model ở tab 'Load & Train' trước!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using TensorFlow & Streamlit | CNN for Vietnamese Food Recognition"
    "</div>",
    unsafe_allow_html=True
)
