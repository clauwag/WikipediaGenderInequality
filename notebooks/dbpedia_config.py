# The DBpedia editions we will consider
MAIN_LANGUAGE = 'en'
LANGUAGES = 'en|bg|ca|cs|de|es|eu|fr|hu|id|it|ja|ko|nl|pl|pt|ru|tr|ar|el'.split('|')

# Where are we going to download the data files
DATA_FOLDER = '/home/egraells/resources/dbpedia'

# Folder to store analysis results
TARGET_FOLDER = './person_results'

# This is used when crawling WikiData.
QUERY_WIKIDATA_GENDER = False
YOUR_EMAIL = 'mail@example.com'