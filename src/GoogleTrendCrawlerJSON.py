# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')



import requests
import json


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

        if not os.path.exists(path):
            os.mkdir(path)
        self.logfilename = path+"log-fails.txt"

        self.html_base = u"http://www.google.com/trends/fetchComponent?q="

        self.query_type = u"&cid=TIMESERIES_GRAPH_0&export=3"
        self.path = path
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}



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

        #people = people.shuffle(axis=0)
        #print people.head()
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




    def crawl_nobelprize_winner(self):

        allpeople = pd.read_csv('data/nobel_identifier_all.csv', delimiter=",", header=None)

        allpeople.columns = ["name", "freebase", "year"]
        print "start crawling %s randomly selected people"%len(allpeople)
        print allpeople.head(n=1)
        print allpeople.shape
        self.run(allpeople, self.path)


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

            #name = name.encode("utf8")
            #name = unicode(name, 'cp1252')

            if os.path.isfile(path+ind+".json"):
                print "found & rename"
                os.rename(path+ind+".json", path+name.replace('/', '')+".json")
            elif os.path.isfile(path+(name.replace('/', ''))+".json"):
                print "found "
                #os.rename(path+(name.replace('/', ''))+".json", path+ind+".json")
            else:
                # make request
                try:
                    #q = u"asdf,qwerty"
                    full_query = self.html_base + name + self.query_type
                    print(full_query)

                    # set header to pretend a user-visit
                    response = requests.get(full_query, headers=self.headers)

                    print(response.status_code)
                    with open(path+name.replace('/', '')+".json", 'w') as outfile:
                        if response.status_code == 200:
                            # no data found
                            outfile.write(response.text.encode("utf8"))
                            outfile.close()

                        elif response.status_code == 203:
                            if response.content.startswith("<!DOCTYPE html>"):
                                # quota limit
                                outfile.close()
                                os.remove(path+name.replace('/', '')+".json")
                                time.sleep(randint(10,30))
                            else:
                                print (response.content)
                                data = json.loads(response.text.encode("utf8"))
                                json.dump(data, outfile)

                except Exception:
                         logfile.write("\n%s"%name)

                # wait a random amount of time between requests to avoid bot detection
                #time.sleep(randint(10,10)) #30



        logfile.close()

if __name__ == "__main__":

    startyear = 1900
    crawler = GoogleTrendCrawler('data/trends-sample-birth'+str(startyear)+'/', startyear)
    crawler.crawl_wikipedia_people(2000, 'data/consolidated_person_data.csv', True)
