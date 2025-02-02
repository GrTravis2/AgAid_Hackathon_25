import cv2
import pandas as pd
import numpy as np
from PIL import Image


def main() -> None:
    colourImg = Image.open("IMG_0937.jpg_part01.jpg")
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(colourImg.size + (3,))
    indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
    allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))


    df = pd.DataFrame(allArray, columns=["y", "x", "red","green","blue"])
    print(df)

if '__name__' == '__main__':
    main()