from configparser import ConfigParser
from duplicate_finder import find_duplicates

# Loads variables from config.ini file
config = ConfigParser()
config.read('config.ini')
path = config.get('main', 'path').replace("\"", "")
outputpath = config.get('main', 'outputpath').replace("\"", "")
accuracy = config.get('main', 'accuracy').replace("\"", "")
chroma_method = config.get('main', 'chroma_method').replace("\"", "")

# runs the system
if outputpath == "":
    outputpath = None

find_duplicates(path, outputpath, accuracy, chroma_method)
