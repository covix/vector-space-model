# Profile-based Information Retrieval
This project assigns incoming document to users based on the users profile. The profile includes several interests, represented as keywords.

## Usage
To run the program, you can either use the provided run script or execute the python srict yourself. The runscript will use a pretrained model and execute it with all the documents provided in the "docs" folder. To run the script by hand, the following arguments can be used:
```
-d or --doc: Path to documents that are supposed to be classified.
-m or --model: If set uses model for classification.
-s or --save: If set saves model after training.
```
For example:
```sh
$ python text_analysis.py -d docs -s
$ python text_analysis.py -d docs -m model_20170425_213040.model
```
