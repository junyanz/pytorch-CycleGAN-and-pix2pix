import glob
import os
import shutil
import subprocess
import heapq

# dummy import for auto refresh
from generator_state import *
from img_utils import create_silhouette, load, and_img

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas


def app():
    # Specify canvas parameters in application
    stroke_width = 3
    stroke_color = '#FFF'
    bg_color = '#000'
    drawing_mode = 'freedraw'
    realtime_update = True

    st.header("CycleGAN image generator")
    st.subheader("Draw a shape")
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if canvas_result is not None:
        img_data = canvas_result.image_data
        if st.button("Generate"):
            im = Image.fromarray(img_data.astype('uint8'), mode="RGBA")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[3])
            background.save("frontend/cyclegan_shapes/shape.png", "PNG")
            subprocess.Popen(['./test_gans.sh'],
                             # stdout=subprocess.DEVNULL,
                             # stderr=subprocess.DEVNULL
                             )

    if st.button("Clear generation"):
        files = glob.glob('frontend/cyclegan_images/*')
        for f in files:
            shutil.rmtree(f)

    images = {'1': [], '2': [], '3': []}

    anding_img = create_silhouette(load('frontend/cyclegan_shapes/shape.png'))
    for dirpath, dirnames, filenames in os.walk("./frontend/cyclegan_images"):
        for image in [os.path.join(dirpath, f) for f in filenames if f.endswith("_fake.png")]:
            path = dirpath.split('/')
            heapq.heappush(images[path[-3][-1]], (int(path[-4]), and_img(load(image), anding_img)))

    if len(images['1']) == 0:
        st.write("Click generate to create images")
    else:
        for i in range(1, 4):
            st.subheader(f'Generation {i}')
            items = [heapq.heappop(images[str(i)]) for _ in range(len(images[str(i)]))]
            if items:
                caps, ims = zip(*items)
                st.image(list(ims), list(map(lambda x: 'epoch: ' + str(x), caps)))


st.set_page_config(
    page_title="Cyclegan image generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

app()