import sys
sys.path.insert(1, "/home/digitalstorm/CVRL Projects/Handwriting Analysis/code")
from os.path import isdir

from siameseLoader import getSiameseAnnotations, getFileNo

dataDir = "code/datasets/siameseRoot"

if isdir(dataDir):
    df = getSiameseAnnotations(dataDir, "test/siameseAnnotations.csv")
    #badRows = df[df.apply(lambda x: getFileNo(x.iloc[0]) == getFileNo(x.iloc[1]))]

else:
    print("false")