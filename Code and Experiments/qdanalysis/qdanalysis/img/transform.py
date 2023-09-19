from numpy import ndarray, array
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skimage import color
from skimage.transform import resize

def imageTransform(images):
    image_transformed = []
    for image in images:
        #not needed
        # image_gray = color.rgb2gray(image)
        image_resized = resize(image, output_shape=(64, 64))
        image_transformed.append(image_resized)

    return image_transformed


def descriptorPipeline():
    return Pipeline([('median impute', SimpleImputer(strategy="median")), ('standard scaler', StandardScaler())])
