"""
contains the code used to transform the given datasets from single images (scanned handwritten documents) to 
    multiple subimages (segmented strokes)
"""

import multiprocessing
import os
from pathlib import Path

"""
CERUG dataset presented in 'Junction Detection in Handwritten Documents and its Application to Writer Identification'
page 1: chinese template text (CERUG-CN)
page 2: chinese freeform (CERUG-CN)
page 3: English template (CERUG-ENG)
page 4: Chinese/English template (CERUG-MIXED)

NOTE: page 3 is split into two parts

CERUG filename format: {Writername}_{pagenumber}-{pagepart}

preparation steps:
1. go throught the folder of the CERUG dataset, for each:
2. create a class based on filename (above)
    * page 1/2 - Writer{num}-CN
    * page 3 (parts 1 and 2) - writer{num}-EN
    * page 4 (optional) - writer{num}-MIXED
3. preprocess image
4. extract stroke images and save
"""

def process_image(filename):
    data = {}
    # extract writername and pagenumber-pagepart from filename
    # Load image
    # Preprocess image
    # Extract strokes
    # Save strokes with class name
    
    pass

def prepare_cerug(input_dir, output_dir):
    
    pool = multiprocessing.Pool()
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        pool.apply_async(process_image, (filepath,))
        
    pool.close()
    pool.join()