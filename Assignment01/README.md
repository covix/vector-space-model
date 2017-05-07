# Profile-based Information Retrieval
This project assigns incoming document to users based on the users profile. The profile includes several interests, represented as keywords.

## Installation
Our implementation is `python2` compatible. The required packages are listed in `requirements.txt` and can be installed with:
```sh
pip2 install -r requirements.txt
```

## Usage
To run the program, you can either use the provided run script or execute the python srict yourself. The runscript will train a model on the 20newsgroup dataset and execute it with all the documents provided in the "docs" folder. 
To run the script by hand, the following arguments can be used:
```
-d or --doc: Path to documents that are supposed to be classified.
-m or --model: If set uses model for classification.
-s or --save: If set saves model after training.
```
For example:
```sh
$ python2 text_analysis.py -d docs -s
$ python2 text_analysis.py -d docs -m model_20170425_213040.model
```

When running the `compute_metrics.py` script, an absolut and a normalized confusion matrics are created. Also evaluation metrics like precision, recall and f1-score will be displayed. To run this a pretrained model needs to be provided. 
For example:
```sh
$ python2 compute_metrics.py model_20170425_213040.model
```

The same holds also for the `plot_tfidf.py` script.

