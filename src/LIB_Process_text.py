import csv
import gzip
import nltk
from nltk.tag import pos_tag, map_tag
from nltk.stem.wordnet import WordNetLemmatizer

#load subjectivity dictionary
fin = open("Tools/subjclueslen1-HLTEMNLP05.tff","rt")
subjdic = dict()
for line in fin:
  fields = line.split(" ")
  contdic = dict()
  for field in fields:
    parts = field.split("=")
    if len(parts) == 2:
      contdic[parts[0]] = parts[1]
  w = contdic["word1"]
  subjdic[w] = contdic["type"], contdic["pos1"], contdic["priorpolarity"].replace("\n", "")
fin.close()


fout = open("textData.dat", "w")
fout.write( "gender" + "\t" + "nwords" + "\t" +  "nadj"  + "\t" +  "nverb"  + "\t" + "nweaksub" + "\t" + "nstrongsub" + "\t" + "npos" + "\t" + "nneg" + "\t" + "nweaksubadj" + "\t" + "nstrongsubadj" + "\t" + "nweaksubverb" + "\t" + "nstrongsubverb" + "\t" + "nposadj" + "\t" + "nnegadj" + "\t" + "nposverb" + "\t" + "nnegverb"  + "\n") 

for i in range(1,5):
	print i
	fin = gzip.open("../data/person_text_"+str(i)+".csv.gz", "rb")
	csvreader = csv.reader(fin, delimiter=',', quotechar='"')
	#uri,wikidata_entity,class,gender,edition_count,available_english,available_editions,birth_year,death_year,page_length,page_out_degree,label,abstract

	for fields in csvreader:
	  gender = fields[3]
	  text = fields[12].decode("utf8").replace("\n", "").replace('"', '')
	  texttokens = nltk.word_tokenize(text)
	  posTagged = pos_tag(texttokens)
	  simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
	  nadj = nverb = nwords =  0
	  nweaksub = nstrongsub = 0
	  npos = nneg = 0
	  nweaksubadj = nstrongsubadj = nweaksubverb = nstrongsubverb = 0
	  nposadj = nnegadj = nposverb = nnegverb = 0

	  for tag in simplifiedTags :
	    if tag[1] != ".":
	      nwords +=1
	      word = tag[0].lower()

	      sv = subjdic.get(word, -1)
	      type = pos = polarity = ""
	      if sv != -1:
	        type= sv[0]
	        pos = sv[1]    # adj verb
	        polarity = sv[2]

	      if type == "weaksubj":
	        nweaksub += 1
	      if type == "strongsubj":
	        nstrongsub += 1
	      if polarity == "positive":
	        npos +=1
	      if polarity == "negative":
	        nneg +=1
	      if polarity == "both":
	        nneg +=1
	        npos +=1

	      if tag[1] == "ADJ":
	        nadj +=1
	        if type == "weaksubj":
	          nweaksubadj += 1
	        if type == "strongsubj":
	          nstrongsubadj += 1
	        if polarity == "positive":
	          nposadj +=1
	        if polarity == "negative":
	          nnegadj +=1
	        if polarity == "both":
	          nnegadj +=1
	          nposadj +=1

	      if tag[1] == "VERB":
	        nverb +=1
	        if type == "weaksubj":
	          nweaksubverb += 1
	        if type == "strongsubj":
	          nstrongsubverb += 1
	        if polarity == "positive":
	          nposverb +=1
	        if polarity == "negative":
	          nnegverb +=1
	        if polarity == "both":
	          nnegverb +=1
	          npos +=1

	  fout.write(gender +"\t" + str(nwords) + "\t" +  str(nadj)  + "\t" +  str(nverb) + "\t" + str(nweaksub) + "\t" + str(nstrongsub) + "\t" + str(npos) + "\t" + str(nneg) + "\t" + str(nweaksubadj) + "\t" + str(nstrongsubadj) + "\t" + str(nweaksubverb) + "\t" + str(nstrongsubverb) + "\t" + str(nposadj) + "\t" + str(nnegadj) + "\t" + str(nposverb) + "\t" + str(nnegverb) + "\n") 

	fin.close()


fout.close()
 
