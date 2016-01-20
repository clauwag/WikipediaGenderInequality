# Gender Asymmetries in Wikipedia

This folder contains Jupyter Notebooks (in Python 2.7, but it could work with Python 3.4) that perform notability, lexical, and network analysis on biographies present in DBpedia and Wikidata, according to gender.

## Requirements 

Install the required libraries in `requirements.txt` (you can use pip).
You also need:

  * [DBpedia Utils](https://github.com/carnby/dbpedia_utils) to iterate over DBpedia data (this is mandatory).
  * [matta](https://github.com/carnby/matta) to generate word clouds of lexical analysis results (optional). 

## Running the Notebooks

First, you need to edit the `dbpedia_config.py` file. This is the current content of the file:

```python
# The DBpedia editions we will consider
MAIN_LANGUAGE = 'en'
LANGUAGES = 'en|bg|ca|cs|de|es|eu|fr|hu|id|it|ja|ko|nl|pl|pt|ru|tr|ar|el'.split('|')

# Where are we going to download the data files
DATA_FOLDER = '/home/egraells/resources/dbpedia'

# Folder to store analysis results
TARGET_FOLDER = '/home/egraells/phd/notebooks/pajaritos/person_results'

# This is used when crawling WikiData.
QUERY_WIKIDATA_GENDER = False
YOUR_EMAIL = 'mail@example.com'
```
  
Its content will change which data is downloaded and analyzed by the notebooks. The notability notebooks take care of downloading and consolidating data.

## Credits

The notebook code in this folder was written by [Eduardo Graells-Garrido](http://carnby.github.io). The notability analysis is original for a paper titled ["Women Through the Glass-Ceiling: Gender Asymmetries in Wikipedia"](http://arxiv.org/abs/1601.04890) with [Claudia Wagner](http://claudiawagner.info/) and [David García](http://dgarcia.eu/). Part of the lexical and network analysis were originally from a paper titled ["First Women, Second Sex: Gender Bias on Wikipedia"](http://arxiv.org/abs/1502.02341) with Mounia Lalmas and Fil Menczer, and were extended for the former paper in these notebooks.