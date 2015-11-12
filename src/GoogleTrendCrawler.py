__author__ = 'wagnerca'

from pytrends.pyGTrends import pyGTrends
import time
from random import randint
import pandas as pd
import os.path
import sys
import codecs
import numpy as np
import re
import urllib2

print sys.stdout.encoding


class GoogleTrendCrawler:

    def __init__(self, path, startyear):
        self.startyear = startyear

        # ADD YOUR ACCOUNT INFOS
        self.google_username = ""
        self.google_password = ""

        if not os.path.exists(path):
            os.mkdir(path)
        self.logfilename = path+"log-fails.txt"

        self.connector = pyGTrends(self.google_username, self.google_password)
        self.path = path



    def create_sample(self, N, allpeople, path):
        # sample 100 people
        allpeople.reset_index(inplace=True)
        people = allpeople.ix[np.random.random_integers(0, len(allpeople), N)]
        people.to_csv(path+"selected_people.csv")
        return people

    def create_stratified_sample(self, N, allpeople, path):
        women = allpeople[allpeople["gender"] == "female"]
        women.reset_index(inplace=True)
        men = allpeople[allpeople["gender"] == "male"]
        men.reset_index(inplace=True)
        print "women and men shape: "
        print women.shape
        print men.shape
        women = women.ix[np.random.random_integers(0, len(women), N)]
        #women.dropnan(inplace=True, how=any)
        men = men.ix[np.random.random_integers(0, len(men), N)]

        print women.shape
        print men.shape
        people = women.append(men)
        print people.head()
        print people.shape

        people.to_csv(path+"selected_people.csv")
        return people



    def crawl_wikipedia_people(self, N, alldatafile, birth_lim):
        allpeople = pd.read_csv(alldatafile, delimiter=",", header=0)
        print allpeople.head(n=5)

        if birth_lim:
            allpeople = allpeople[(allpeople.birth_year >= self.startyear) & (allpeople.birth_year <= 2000)]

        print "start crawling %s randomly selected people"%N
        if os.path.isfile(self.path+"selected_people.csv"):
            print "use selected_people file that lies here %s"%self.path
            people = pd.DataFrame.from_csv(self.path+"selected_people.csv")
        else:
            print "select people"
            people = self.create_sample(N, allpeople, self.path)

        self.run(people, self.path)



    def crawl_strata_wikipedia_people(self, N, alldatafile, birthstrata, minmax_edition):
        allpeople = pd.read_csv(alldatafile, delimiter=",", header=0)
        print allpeople.head(n=5)

        if birthstrata:
            allpeople = allpeople[(allpeople.birth_year >= 1900) & (allpeople.birth_year <= 2000)]

        if len(minmax_edition) == 2:
            min = minmax_edition[0]
            max = minmax_edition[1]
            allpeople = allpeople[(allpeople.edition_count >= min) & (allpeople.edition_count <= max)]


        print "start crawling %s randomly selected people"%N
        if os.path.isfile(self.path+"selected_people.csv"):
            print "use selected_people file that lies here %s"%self.path
            people = pd.DataFrame.from_csv(self.path+"selected_people.csv")
        else:
            print "select people"
            people = self.create_stratified_sample(N, allpeople, self.path)

        self.run(people, self.path)





    def run(self, people, path):
        logfile = open(self.logfilename, 'w+')

        print people.head()
        for ind,vals in people.iterrows():
            #print vals
            name =  vals["label"]
            ind = str(vals["index"])
            print name
            print ind

            if str(name).lower() == "nan":
                continue

            if "(" in name:
                #remove additional info from name
                pos = name.find("(")
                name = name[0:pos]

            # remove the letter-dot stuff
            name = re.sub(r'\s+[A-Z]\.\s+', ' ', name)
            # remove quoted stuff e.g. Katharina "kate" Mcking
            name = re.sub(r'\s+"[A-Za-z\s]+"\s+', ' ', name)
            # remove stuff after the comma e.g. James Dean, King from Sweden
            name = re.sub(r',\s*.+', ' ', name)

            if os.path.isfile(path+ind+".csv"):
                print "found & rename"
                os.rename(path+ind+".csv", path+name.replace('/', '')+".csv")
            elif os.path.isfile(path+(name.replace('/', ''))+".csv"):
                print "found "
            else:
                # make request
                try:
                    self.connector.request_report([name])
                except UnicodeEncodeError:

                    try:
                        print "encoding problems with encoding of name %s" % name.encode('cp1252')
                        self.connector.request_report([name.encode('cp1252')]) #, "book"
                    except UnicodeEncodeError:
                         logfile.write("failed 1 + %s"%name)
                except urllib2.HTTPError, e:
                    print e.code
                except urllib2.URLError, e:
                    print e.args


                # wait a random amount of time between requests to avoid bot detection
                time.sleep(randint(10,70))

                # download file
                try:
                    self.connector.save_csv(path, name.replace('/', '')) #name.replace('/', ''))
                except UnicodeEncodeError:
                    try:
                        self.connector.save_csv(path, name.replace('/', '')).encode('cp1252')# (name.replace('/', ''))
                    except UnicodeEncodeError:
                        logfile.write("failed 2: %s"%name)

                except UnicodeDecodeError:
                    try:
                        reload(sys)
                        sys.setdefaultencoding('cp1252')
                        print "decoding problems when saving file %s" % (ind).decode('cp1252').encode('cp1252')
                        #connector.save_csv(path, name.encode('cp1252'))
                        self.connector.save_csv(path, (name.replace('/', '')).decode('cp1252'))
                    except UnicodeDecodeError:
                        print "UnicodeDecodeError"
                    except UnicodeEncodeError:
                         logfile.write("failed 3: %s"%name)

        logfile.close()



if __name__ == "__main__":
    startyear = 1900
    crawler = GoogleTrendCrawler('data/trends-sample-birth'+str(startyear)+'/', startyear)
    crawler.crawl_wikipedia_people(2000, 'data/consolidated_person_data.csv', True)
