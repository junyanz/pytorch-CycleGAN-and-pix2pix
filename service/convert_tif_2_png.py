from glob import glob
from PIL import Image
import rasterio as rio
import numpy as np


for fn in glob("./data/train/hotsat1/*"):
    Image.open(fn).save(fn.replace("data", "imgs").replace(".tif", ".png"))

for fn in glob("./data/train/sentinel2/*"):
    with rio.open(fn) as fp:
        data = (fp.read(1) * 10000).astype(np.uint16)
    Image.fromarray(data).save(fn.replace("data", "imgs").replace(".tiff", ".png"))
