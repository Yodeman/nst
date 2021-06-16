import streamlit as st
import numpy as np
from utils import *

def main():
    st.title("Welcome to YourArt 📸")
    st.header("Turn your picture into an art style of your choice.")
    st.write(
        """This is a demonstration of my practice on neural style transfer,
            and hosting model on google cloud platform. Neural style transfer is
            an application of computer vision to generate a new picture from
            two pictures (the original image and the style image). Below is an
            example of what this does.
        """
    )
    pic1 = resize("./mary_03.jpg")
    pic2 = resize("./drop-of-water.jpg")
    pic3 = resize("./mary_drop_of_water.jpg")
    col1, col2, col3 = st.beta_columns(3)
    with col1: st.image(pic1, width=WIDTH, caption="Original Image")
    with col2: st.image(pic2, width=WIDTH, caption="Style Image") 
    with col3: st.image(pic3, width=WIDTH, caption="Generated Image") 
    #st.write('Click "Generate Image" on the sidebar to start.')
    st.write("""
        Upload the content image and style image to start.
        Uploaded images will be cropped if images are not square images.
        """
    )
    ##st.sidebar.title("What to do?")
    ##app_mode = st.sidebar.selectbox(
    ##    "Choose the app mode", ["Home page", "Generate Image"]
    ##)
    ##if app_mode == "Home page":
    ##    st.sidebar.success('To continue select "Generate Image"')
    ##elif app_mode == "Generate Image":
    ##   generate_image()
    file1 = st.file_uploader("Upload original/content image")
    if file1:
        f1_img = resize(file1) #Image.open(file1)
        st.image(f1_img, width=WIDTH, caption="your content image")
    file2 = st.file_uploader("Upload style image")
    if file2:
        f2_img = resize(file2) #Image.open(file2)
        st.image(f2_img, width=WIDTH, caption="your style image")
    if st.button("Generate Image"):
        if file1 and file2:
            generate_image(file1, file2)
        else:
            st.warning("Ensure you upload both content and style image...😒")
        
def generate_image(img_1, img_2):
    st.success("Your art is being generated wait for few seconds, good stuff sometimes takes time...😉")
    output = nst(img_1, img_2, None, 500)
    output = tensor_pil(output)
    st.image(output, width=300, caption="your art.")

def tensor_pil(nst_img):
    n_image = nst_img.cpu().clone()
    n_image = n_image.squeeze(0)
    n_image = unloader(n_image)
    return n_image

if __name__ == "__main__":
    WIDTH = 200
    main()
