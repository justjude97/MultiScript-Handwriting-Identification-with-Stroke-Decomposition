# Multi-Script Writer Identification by Fragmenting Strokes
This is the work so far for my masters thesis on Handwriting Identification/Verification in the Multi-Script setting. Writing Identification deals with the identificaiton of one or more writers, and Multi-Script Writing Identification deals with Writing Identification where a writer can create handwritten documents in one or more writing languages.

requirements.txt contains the list of python modules required to set up the environment for this project (note that the submodules qdanalysis and sknw may have to be installed separately as they are local packages)

CERUG dataset pulled from: https://github.com/shengfly/writer-identification

## sknw_mswi
skeleton network mswi is a simple fork of the skeleton network library https://github.com/Image-Py/sknw. The only difference between the submodule and its form is that the numba calls are removed to prevent some compatability issues. 