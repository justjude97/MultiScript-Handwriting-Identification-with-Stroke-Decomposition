#!/bin/bash

usage="splice_signature_page.sh [signature page] [thesis] [output filename] -- program to splice the completed signature page into a thesis"

pdftk A=$1 B=$2 cat A1 B2-end output $3
