from qdanalysis.img import getData, combineData

import numpy as np
import pandas as pd

annotations_file = "/home/digitalstorm/CVRL Projects/Handwriting Analysis/siamese_annotations_raw_sameword.csv"
image_file_path = "/home/digitalstorm/CVRL Projects/Handwriting Analysis/datasets/siameseRoot"

annotations = pd.read_csv(annotations_file)
mapping, images = getData.getImages(image_file_path)
descriptors = getData.getDescriptors(images)

test = annotations.head(20)
test_descriptors = combineData.combineDescriptors(test, descriptors, mapping, 5)