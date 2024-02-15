"""
contains the code used to transform the given datasets from single images (scanned handwritten documents) to 
    multiple subimages (segmented strokes)

current strategy is to split dataset into multiple datasets dependant on script, and combine as needed.

e.g:
CERUG -> EN, CN, MIXED
"""

from multiprocessing import Process, Lock
import os
from pathlib import Path
import cv2 as cv

import qdanalysis.strokedecomposition as sd
import qdanalysis.preprocessing as prep

"""
    perform the stroke decomposition process on the given batches of files
    * file_batch is now a list of tuples of the form (filepath, writing script)
"""
def process_batch(class_name, file_batch, base_output_dir, logging_lock):
    
    for file, script in file_batch:
        #make directory for writer in specific script folder
        output_dir = os.path.join(base_output_dir, script, class_name)
        os.makedirs(output_dir, exist_ok=True)

        #danger zone
        try:
            # Process each file
            image = cv.imread(file, cv.IMREAD_GRAYSCALE)
            processed_strokes = sd.simple_stroke_segment(image) # processing logic for each file

        #if something happens when trying to load or process a file, log and skip
        except Exception as e:
            logging_lock.acquire()
            
            print(f"exception occured when processing file: {file}")
            print(e)

            logging_lock.release()

            continue

            
        for image_no, subimage in enumerate(processed_strokes):
            # Save the processed data
            #TODO: ugly. fix later. also figure out ideal image format
            output_path = os.path.join(output_dir, f'{Path(file).stem}_{image_no}.png')
            cv.imwrite(output_path, subimage)

"""
CERUG dataset presented in 'Junction Detection in Handwritten Documents and its Application to Writer Identification'
page 1: chinese template text (CERUG-CN)
page 2: chinese freeform (CERUG-CN)
page 3: English template (CERUG-ENG)
page 4: Chinese/English template (CERUG-MIXED)

NOTE: page 3 is split into two parts

CERUG filename format: {Writername}_{pagenumber}-{pagepart}
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
    
        return writer_class, writer_script

    #create subdirectories for each writing script
    writing_scripts = ['CN', 'EN', 'MIXED']
    for script in writing_scripts:
        script_subdir = os.path.join(output_dir, script)
        os.makedirs(script_subdir, exist_ok=False)

    # Organize files into batches by class
    file_batches = {}
    for file_path in Path(input_dir).iterdir():
        class_name, script = get_class_from_filename(file_path.stem)
        
        file_path = str(file_path)
        if class_name not in file_batches:
            file_batches[class_name] = []
        
        #filepath wrapped up with script so processes can decide which folder to put a output
        file_batches[class_name].append((file_path, script))

        
    logging_lock = Lock()

    # Main process
    processes = []
    for class_name, batch in file_batches.items():
        p = Process(target=process_batch, args=(class_name, batch, output_dir, logging_lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    prepare_cerug("./experimentation/CERUG", "./experimentation/output/cerug_test_processing")