import streamlit as st
import pandas as pd
import streamlit.components.v1 as c
import variables as v
from PIL import Image
import cv2
from tensorflow.keras.models import load_model


import sys
sys.path.append(r'C:\\Users\\javie\\OneDrive\\Documentos\\Hacker\\Proyect_ML')

from src.evaluation import evaluate_img

# Ass visual files

st.set_page_config(page_title="PROYECTO MACHINE LEARNING",
                   page_icon="favicon.ico")


st.title("¿Podemos predecir el tiempo solo con imágenes?")

# Initialize sessions_states
if 'subtitle' not in st.session_state:
    st.session_state['subtitle'] = ''
elif 'desc' not in st.session_state:
    st.session_state['desc'] = ''
    st.session_state['subtitle'] = ''
elif 'an' not in st.session_state:
    st.session_state['an'] = ''
elif 'img_pred' not in st.session_state:
    st.session_state['img_pred'] = None


img_port = Image.open(v.URL_IMG_PORT)

st.image(img_port, width=750 )


# Using object notation
ds = st.sidebar.button("Análisis y conclusiones técnicas")
buss = st.sidebar.button("Análisis y conclusiones para negocio")

if ds:
    st.session_state.subtitle = "<h2>Análisis y conclusiones técnicas</h2>"
    st.session_state.desc = "<p>Se realizó un análisis para poder tratar nuestro problema. Este análisis se ha basado en el estudio de un  dataset con 11 tipos de fotos de fenómenos atmosféricos.</p>"
    an = \
    """<p> El dataset se distribuye: </p>
                    <ul>
                        <li>rime: 1160 </li>
                        <li>fogsmog: 850 </li>
                        <li>dew: 698 </li>
                        <li>sandstorm: 692 </li>
                        <li>glaze: 639 </li>
                        <li>snow: 620 </li>
                        <li>hail: 591 </li>
                        <li>rain: 526 </li>
                        <li>frost: 475 </li>
                        <li>lightning: 377 </li>
                        <li>rainbow: 232 </li>
                    </ul>
    """
    an2 = " <p> Tras leer las imagenes y clasificarlas, se rescalaron a 50x50 pixeles, aplicamos un reducción en su complejidad usando un PCA (Principal Component Analysis) para solo usar 50 pixeles como máximo y se almacenó en un dataset.</p>"
    an3 = \
    """ <p> Con el dataset con nuestros pixeles, entrenamos los siguientes modelos: </p>
        <h4>Modelos supervisados</h4>
        <ul>
            <li>Logistic Regression </li>
            <li>Random Forest</li>
            <li>Support Verctor Machine</li>
            <li>K-Neighbors Classifier</li>
            <li>Decision Tree</li>
        </ul>
        <p> Evaluamos los modelos y conseguimos las siguientes precisiones:</p>
    """
    an4 = "<p> Apreciamos que el mejor modelo es el Random Forest con un 63% de precisión. \n</p> <p>Podemos mejorar la precisión usando redes neuronales, en concreto Redes convolucionales. Creamos un modelo con la siguiente arquitectura:</p>"
    an5 = "<p> Después de entrenar a varios modelos CNN, encontramos un modelo bastante significativo, cuyo entrenamiento fue el siguiente: </p>"
    an5_5 = "<p> Vemos en su matriz de confusión como predice mi modelo según las clases: "
    an6 = "<p> Como bien marca el grafico, con este modelo conseguimos un 80% de precisión en la época 49, que es el modelo que mejor se ha adaptado a nuestros datos. \n</p> <p>¡Ahora prueba tu! Prueba a subir una foto entre las clases que pueda predecir y asi lo compruebas tu mismo.</p>"

    img_dew = Image.open(v.URL_IMG_DEW)
    img_fogsmog = Image.open(v.URL_IMG_FOGSMOG)
    img_frost = Image.open(v.URL_IMG_FROST)
    img_glaze = Image.open(v.URL_IMG_GLAZE)
    img_hail = Image.open(v.URL_IMG_HAIL)
    img_lightning = Image.open(v.URL_IMG_LIGHTNING)
    img_rain = Image.open(v.URL_IMG_RAIN)
    img_rainbow = Image.open(v.URL_IMG_RAINBOW)
    img_rime = Image.open(v.URL_IMG_RIME)
    img_sandstorm = Image.open(v.URL_IMG_SANDSTORM)
    img_snow = Image.open(v.URL_IMG_SNOW)

    st.markdown(st.session_state.subtitle, unsafe_allow_html=True)
    st.markdown(st.session_state.desc, unsafe_allow_html=True)

    firts_co , cent_co, last_co = st.columns(3)

    with firts_co:
        st.image(img_dew, width=250)
        st.image(img_glaze, width=250)
        st.image(img_rain, width=250)
    with cent_co:
        st.image(img_fogsmog, width=250)
        st.image(img_hail, width=250)
        st.image(img_rainbow, width=250)
    with last_co:
        st.image(img_frost, width=250)
        st.image(img_lightning, width=250)
        st.image(img_sandstorm, width=250)

    st.write("\n\n")

    firts_co , last_co = st.columns(2)

    with firts_co:
        st.markdown(an, unsafe_allow_html=True)
    with last_co:
        img_hist = Image.open(v.URL_IMG_HIST)
        st.image(img_hist, width=350)

    st.markdown(an2, unsafe_allow_html=True)

    with st.expander("Tabla pixeles"):
            st.write("Los datos de los pixeles con su tipo de imagen")
            #df = pd.read_csv(v.URL_TRAIN_CSV, sep=";")
            #df.drop(columns="Unnamed: 0", inplace=True)
            #st.write(df.head(10))

    with st.expander("PCA"):
            st.write("Análisis PCA")
            img_pca_comp = Image.open(v.URL_IMG_PCA_COMP)
            img_pca_acum = Image.open(v.URL_IMG_PCA_ACUM)

            firts_co , last_co = st.columns(2)
            with firts_co:
                st.image(img_pca_comp, width=380)
            with last_co:
                st.image(img_pca_acum, width=380)

    st.markdown(an3, unsafe_allow_html=True)

    df = pd.read_csv(v.URL_SM_ACC_CSV, sep=",")
    df.drop(columns=["Unnamed: 0","name"], inplace=True)
    model_names = ["Logistic Regression","Random Forest","Support Verctor Machine","K-Neighbors Classifier", "Decision Tree"]
    df["name_model"] = model_names
    df.set_index('name_model', inplace=True)
    st.write(df.head(5))

    st.markdown(an4, unsafe_allow_html=True)

    with st.expander("Modelo CNN"):
        firts_co , cent_co, last_co = st.columns(3)
        st.write("Arquitectura CNN")
        with cent_co:
            img_cnn = Image.open(v.URL_MODEL_CNN)
            st.image(img_cnn, width=380)

    st.markdown(an5, unsafe_allow_html=True)

    with st.expander("Model history"):
        st.write("Historial aprendizaje")
        img_cnn = Image.open(v.URL_HIST_MODEL_CNN)
        st.image(img_cnn, width=675)

    st.markdown(an5_5, unsafe_allow_html=True)
    with st.expander("Matriz de confusión"):
        img_cm = Image.open(v.URL_CONFUSION_MATRIX)
        st.image(img_cm, width=550)

    st.markdown(an6, unsafe_allow_html=True)

elif buss:
    st.session_state.subtitle = "<h2>Análisis y conclusiones para negocio</h2>"

    st.markdown(st.session_state.subtitle, unsafe_allow_html=True)

st.session_state.img_pred = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if st.session_state.img_pred is not None:
    # Mostrar la imagen cargada
    image = Image.open(st.session_state.img_pred) 
    st.image(image)

    final_model = load_model(v.URL_FINAL_MODEL)
    class_names = ['dew',
                    'fogsmog',
                    'frost',
                    'glaze',
                    'hail',
                    'lightning',
                    'rain',
                    'rainbow',
                    'rime',
                    'sandstorm',
                    'snow']
    predictions, predicted_label_name = evaluate_img(image,final_model,class_names)

    firts_co , last_co = st.columns(2)

    with firts_co:
        st.markdown(f"La imagen analizada es: <b>{predicted_label_name}</b>", unsafe_allow_html=True)


    class_names_encode = {i:c for i,c in enumerate(class_names)}
    df = pd.DataFrame(predictions)
    df.rename(columns=class_names_encode, inplace=True)
    st.write("Las probabilidades de que sea cada clase son: ", df)







