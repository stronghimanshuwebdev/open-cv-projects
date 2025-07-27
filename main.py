import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime

st.set_page_config(layout="wide", page_title="OpenCV Image Editor")
st.markdown("""
    <style>
    .main {background-color: #f7f7f9;}
    h1 {color: #1f4e79;}
    .stButton button {background-color: #1f4e79; color: white; border-radius: 6px;}
    .stSelectbox, .stSlider, .stDownloadButton {margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("End-to-End Image Editing App using OpenCV")

# Sidebar Controls
st.sidebar.header("Image Controls")

uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    section = st.sidebar.radio("Choose Section", ["Basic Processing", "Color and Filtering", "Contours & Morphology"])
    final_output = None

    if section == "Basic Processing":
        width = st.sidebar.slider("Width", 50, 1000, image.shape[1])
        height = st.sidebar.slider("Height", 50, 1000, image.shape[0])
        resized = cv2.resize(image, (width, height))

        crop_x = st.sidebar.slider("Crop X", 0, width - 1, 0)
        crop_y = st.sidebar.slider("Crop Y", 0, height - 1, 0)
        crop_w = st.sidebar.slider("Crop Width", 1, width - crop_x, width)
        crop_h = st.sidebar.slider("Crop Height", 1, height - crop_y, height)
        cropped = resized[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        rotate = st.sidebar.selectbox("Rotate", ["None", "90 Clockwise", "90 Counterclockwise", "180"])
        if rotate == "90 Clockwise":
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == "90 Counterclockwise":
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == "180":
            cropped = cv2.rotate(cropped, cv2.ROTATE_180)

        flip = st.sidebar.selectbox("Flip", ["None", "Horizontal", "Vertical"])
        if flip == "Horizontal":
            cropped = cv2.flip(cropped, 1)
        elif flip == "Vertical":
            cropped = cv2.flip(cropped, 0)

        final_output = cropped
        st.image([image, cropped], caption=["Original", "Processed"], width=300)

    elif section == "Color and Filtering":
        colspace = st.sidebar.selectbox("Color Space", ["RGB", "Grayscale", "HSV", "LAB"])
        if colspace == "Grayscale":
            filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        elif colspace == "HSV":
            filtered = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif colspace == "LAB":
            filtered = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        else:
            filtered = image

        blur_type = st.sidebar.selectbox("Blur Type", ["None", "Gaussian", "Median", "Bilateral"])
        ksize = st.sidebar.slider("Kernel Size", 1, 25, 5, step=2)
        if blur_type == "Gaussian":
            filtered = cv2.GaussianBlur(filtered, (ksize, ksize), 0)
        elif blur_type == "Median":
            filtered = cv2.medianBlur(filtered, ksize)
        elif blur_type == "Bilateral":
            filtered = cv2.bilateralFilter(filtered, ksize, 75, 75)

        final_output = filtered
        st.image([image, filtered], caption=["Original", f"Filtered ({blur_type})"], width=300)

    elif section == "Contours & Morphology":
        mode = st.sidebar.selectbox("Select Processing Mode", ["Thresholding", "Edge Detection", "Morphology", "Contours"])

        if mode == "Thresholding":
            threshold_type = st.sidebar.selectbox("Threshold Type", ["Binary", "Adaptive", "Otsu"])
            if threshold_type == "Binary":
                thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
                _, out = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), thresh_val, 255, cv2.THRESH_BINARY)
            elif threshold_type == "Adaptive":
                out = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        elif mode == "Edge Detection":
            edge_type = st.sidebar.selectbox("Edge Method", ["Sobel", "Laplacian", "Canny"])
            if edge_type == "Sobel":
                out = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=ksize)
            elif edge_type == "Laplacian":
                out = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            else:
                low = st.sidebar.slider("Canny Low Threshold", 0, 255, 100)
                high = st.sidebar.slider("Canny High Threshold", 0, 255, 200)
                out = cv2.Canny(image, low, high)
            out = cv2.convertScaleAbs(out)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        elif mode == "Morphology":
            morph_op = st.sidebar.selectbox("Morph Operation", ["Dilation", "Erosion", "Opening", "Closing"])
            kernel_size = st.sidebar.slider("Kernel Size", 1, 20, 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            if morph_op == "Dilation":
                out = cv2.dilate(bin_img, kernel, iterations=1)
            elif morph_op == "Erosion":
                out = cv2.erode(bin_img, kernel, iterations=1)
            elif morph_op == "Opening":
                out = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
            else:
                out = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        elif mode == "Contours":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out = image.copy()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(out, (int(cx), int(cy)), int(radius), (0, 255, 255), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        final_output = out
        st.image([image, out], caption=["Original", f"{mode} Output"], width=300)

    if final_output is not None:
        st.subheader("Download Processed Image")
        is_success, buffer = cv2.imencode(".png", final_output)
        if is_success:
            st.download_button(
                label="Download Image",
                data=buffer.tobytes(),
                file_name=f"edited_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
