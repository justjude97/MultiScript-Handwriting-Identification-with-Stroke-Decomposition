"""
contains the code used to transform the given datasets from single images (scanned handwritten documents) to 
    multiple subimages (segmented strokes)
"""

from multiprocessing import Process

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

#TODO: modify to receive classname from function, ro method to extract classname from function
def process_batch(file_batch, base_output_dir):

    #TODO
    class_name = None
    
    output_dir = os.path.join(base_output_dir, class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for file in file_batch:
        # Process each file
        processed_data = ... # processing logic for each file
        # Save the processed data
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(file)}")
        processed_data.save(output_path)

def prepare_cerug(input_dir, output_dir):
    
    def get_class_from_filename(filename):
        # Logic to extract class from filename, e.g., 'Writer0101_01' -> 'Writer0101'
        return filename.split('_')[0]

    # Organize files into batches by class
    file_batches = {}
    for file in all_files:
        class_name = get_class_from_filename(file)
        if class_name not in file_batches:
            file_batches[class_name] = []
        file_batches[class_name].append(file)

    # Main process
    processes = []
    for batch in file_batches.values():
        p = Process(target=process_batch, args=(batch, "base_output_directory"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    print("hello world!")