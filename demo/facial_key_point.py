import streamlit as st
from PIL import Image, ImageDraw
from src.facial_key_point.utils.facial_key_point_detection import FacialKeyPointDetection
from io import BytesIO

facial_key_point_detection = FacialKeyPointDetection()

st.set_page_config(page_title="Facial Key Point Detection", page_icon=":camera:", layout="wide")

st.sidebar.title("Settings")
point_radius = st.sidebar.slider("Select Key Point Radius", 1, 10, 2)
point_color = st.sidebar.color_picker("Choose Key Point Color", "#FF0000")

st.markdown(
    "<h1 style='text-align: center; color: #3498db;'>Facial Key Point Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload a facial image to detect key points.</p>",
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader('Choose an image (jpg, jpeg, png)', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #2ecc71;'>Original Image</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)
    
    with col2:
        st.markdown("<h3 style='text-align: center; color: #e74c3c;'>Processed Image</h3>", unsafe_allow_html=True)
        
        with st.spinner("Detecting key points..."):
            _, kp = facial_key_point_detection.predict(image)
        
        draw = ImageDraw.Draw(image)
        for x, y in zip(kp[0], kp[1]):
            draw.ellipse(
                [(int(x.item()) - point_radius, int(y.item()) - point_radius),
                 (int(x.item()) + point_radius, int(y.item()) + point_radius)],
                fill=point_color
            )
        
        st.image(image, caption="Processed Image with Key Points", use_column_width=True, width=300)

    st.markdown("<br>", unsafe_allow_html=True)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    st.download_button(
        label="Download Processed Image", 
        data=buffered,
        file_name="processed_image.png", 
        mime="image/png"
    )
