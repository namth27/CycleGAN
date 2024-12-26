import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='CycleGAN')
class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

  def get_config(self):
    return {"epsilon": self.epsilon}

  @classmethod
  def from_config(cls, config):
    return cls(**config)

# Load CycleGAN model
@st.cache_resource
def load_model(model_path):
    # Đường dẫn tới model đã train
    model = tf.keras.models.load_model(model_path, custom_objects={"InstanceNormalization": InstanceNormalization})
    return model



# Đường dẫn tới các model đã train
model_paths = {
    "Chuyển đổi phong cảnh theo Van Gogh": "gen_f_vangoh_4.keras",
    "Chuyển đổi chân dung người theo Anime": "gen_g.keras"
}

models = {key: load_model(path) for key, path in model_paths.items()}


# Hàm xử lý ảnh đầu vào
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)  # Resize ảnh
    image_array = np.array(image)  # Convert thành numpy array
    image_array = (image_array / 127.5) - 1.0  # Chuẩn hóa về [-1, 1]
    return np.expand_dims(image_array, axis=0)  # Thêm batch dimension


# Hàm postprocess ảnh đầu ra
def postprocess_image(image):
    image = (image + 1.0) * 127.5  # Chuẩn hóa về [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)  # Đảm bảo giá trị trong khoảng [0, 255]
    return Image.fromarray(image[0])  # Chuyển về PIL Image

# Giao diện Streamlit
st.title("Demo CycleGAN - Chuyển đổi phong cách ảnh")
st.write("Ứng dụng chuyển đổi phong cách ảnh sử dụng model CycleGAN.")

# Step 1: User selects model
style_option = st.selectbox("Chọn kiểu chuyển đổi:", list(model_paths.keys()))

# Step 2: User uploads image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Đọc file tải lên
        original_image = Image.open(uploaded_file)
        original_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh tải lên: {e}")
        st.stop()

    # Hiển thị ảnh gốc
    st.image(original_image, caption="Ảnh gốc")

    # Step 3: On submit, perform style transfer
    if st.button("Submit"):
        with st.spinner("Đang xử lý..."):
            try:
                # Load the selected model
                model = load_model(model_paths[style_option])

                # Xử lý ảnh đầu vào
                input_image = preprocess_image(original_image)

                # Dự đoán ảnh đầu ra
                output_image = model.predict(input_image)

                # Xử lý ảnh đầu ra
                output_image = postprocess_image(output_image)

                # Resize ảnh gốc để khớp với ảnh đầu ra
                resized_original = original_image.resize(output_image.size)

                #Hiển thị kết quả
                st.write("Kết quả:")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized_original, caption="Ảnh gốc")
                with col2:
                    st.image(output_image, caption="Ảnh sau chuyển đổi")

                    # Tải xuống ảnh
                    buffer = BytesIO()
                    output_image.save(buffer, format="JPEG")
                    buffer.seek(0)

                    st.download_button(
                        label="Tải xuống ảnh đã chuyển đổi",
                        data=buffer,
                        file_name=f"transformed_{style_option.replace(' ', '_')}.jpg",
                        mime="image/jpeg"
                    )
            except Exception as e:
                st.error(f"Lỗi trong quá trình xử lý: {e}")





