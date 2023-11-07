from pathlib import Path
import pandas as pd
from itertools import combinations
from os.path import relpath

def getFileNo(filePath: str):
    """
        helper function to return the assigned label of the word image
        may change in the future so I abstracted it into its own function
    """
    return filePath.split('/')[-1].split('_')[0]

def getParentDir(filePath: str):
    return filePath.split("/")[-2]
    
def getImgList(srcDir: str):
    """
        returns a list of every image contained within a directory and every subdirectory within that directory
    """

    imgFileList = pd.Series([], dtype="string")

    with Path(srcDir) as root:
        #loop through every subdirectory and get every combination
        for wordDir in root.iterdir():
            if(wordDir.is_dir()):
                #TODO check if file is an image, or not
                imgFileList = imgFileList.append(pd.Series([x.as_posix() for x in wordDir.iterdir()], dtype="string"), ignore_index = True)

    return imgFileList

def getSiameseAnnotations(srcDir : str, fileName = None, sameWord = True):
    #list of images on the left and right sides of our combinations
    leftIm = []
    rightIm = []
    #label dictates whether the images were written by the same writer, or not
    label = []

    imgList = getImgList(srcDir)

    for comb in combinations(imgList, 2):
        
        leftWord = comb[0]
        rightWord = comb[1]

        #skip the current combination if the sameWord flag is true and the directories are different
        if sameWord:
            leftDir = getParentDir(leftWord)
            rightDir = getParentDir(rightWord)
        
            if leftDir != rightDir:
                continue
        
        leftWord = relpath(leftWord, srcDir)
        rightWord = relpath(rightWord, srcDir)

        leftIm.append(leftWord)
        rightIm.append(rightWord)
        
        #now generate and append the label
        #writer of the left image
        leftLabel = getFileNo(leftWord)
        #writer of the right image
        rightLabel = getFileNo(rightWord)
    
        #1 if the writer of both images is the same person, zero otherwise
        label.append(1 if leftLabel == rightLabel else 0)

    #generate siamese dataset annotations as csv file
    df = pd.DataFrame({'leftIm': leftIm, 'rightIm': rightIm, 'label': label})
    df.sort_values("label", inplace=True)
    
    if fileName is not None:
        df.to_csv(fileName, index=False)

    #decided to return the dataframe no matter what
    return df
