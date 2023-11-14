#!/bin/bash
doc_directory=docs
mkdir -p $doc_directory
pdoc ./src/linearwavetheory -o $doc_directory --math
