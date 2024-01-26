"""
contains the code used to transform the given datasets from single images (scanned handwritten documents) to 
    multiple subimages (segmented strokes)
"""

from multiprocessing import Process
import os
from pathlib import Path
import cv2 as cv

import qdanalysis.strokedecomposition as sd
import qdanalysis.preprocessing as prep

#TODO: modify to receive classname from function, ro method to extract classname from function
def process_batch(class_name, file_batch, base_output_dir):
    
    output_dir = os.path.join(base_output_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for file in file_batch:
        # Process each file
        image = cv.imread(file)
        processed_strokes = sd.simple_stroke_segment(image) # processing logic for each file

        for image_no, subimage in enumerate(processed_strokes):
            # Save the processed data
            #TODO: ugly. fix later. also figure out ideal image format
            output_path = os.path.join(output_dir, f'{Path(file).stem}_{image_no}.png')
            cv.imwrite(output_path, subimage.astype(int)*255)

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
def prepare_cerug(input_dir, output_dir):
    
    #reformat to {writerclass}_{writerscript}
    def get_class_from_filename(filename):
        #script part is unimportant
        # {writerclass}_{script class}-{script part}
        writer_split = filename.split('_')
        writer_class = writer_split[0]

        page_no = writer_split[1].split('-')[0]
        
        if page_no == '01' or page_no == '02':
            writer_script = 'CN'
        elif page_no == '03':
            writer_script = 'EN'
        else:
            writer_script = 'MIXED'
    
        return writer_class + '_' + writer_script

    # Organize files into batches by class
    file_batches = {}
    for file_path in Path(input_dir).iterdir():
        class_name = get_class_from_filename(file_path.stem)
        file_path = str(file_path)
        if class_name not in file_batches:
            file_batches[class_name] = []
        file_batches[class_name].append(file_path)

    # Main process
    processes = []
    for class_name, batch in file_batches.items():
        p = Process(target=process_batch, args=(class_name, batch, output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    prepare_cerug("./experimentation/CERUG", "./experimentation/output/cerug_test_processing")