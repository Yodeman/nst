import streamlit as st
import numpy as np
from PIL import Image
import googleapiclient.discovery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "paul-nst-project-317322-7ddf893d15e6.json"
PROJECT = "paul-nst-project" 
REGION = "us-central1" 
MODEL = "nst_model_0"

def resize(img):
    img = Image.open(img)
    dim = min(img.size)
    n_img = img.crop((0, 0, dim, dim))
    return n_img
    
def make_prediction(c_img, s_img):
    cont_img = np.array(resize(c_img)).tolist()
    style_img = np.array(resize(s_img)).tolist()
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.expand_dims(image, axis=0)
    prediction = predict_json(project=PROJECT,
                         region=REGION,
                         model=MODEL,
                         instances=[cont_img, style_img],
                         version="v001")
    pred = Image.fromarray(np.array(prediction))
    return pred
def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'data': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
  

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
    with col1:
        st.image(pic1, width=WIDTH, caption="Original Image")
    st.write(" + ")
    with col2:
        st.image(pic2, width=WIDTH, caption="Style Image") 
    st.write(" = ")
    with col3: 
        st.image(pic3, width=WIDTH, caption="Generated Image") 
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
            generate_image(f1_img, f2_img)
        
def generate_image(img_1, img_2):
    output = make_prediction(img_1, img_2)
    #output = tensor_pil(output)
    st.image(output, width=300, caption="your art.")

if __name__ == "__main__":
    WIDTH = 200
    main()