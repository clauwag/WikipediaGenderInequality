# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')
from os import listdir
from os.path import isfile, join
import scipy.stats as stats
import numpy as np
import pandas as pd
import pylab as plt
from  scipy.stats import itemfreq
import sys
import util as ut
import re
import os
import seaborn as sns
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
from statsmodels.api import families
from numpy import log
import datetime, locale
import json

class GoogleTrendAnalyzer:
    """
        Analyze Google Trend Output
        Google Trends adjusts search data to make comparisons between terms easier. Otherwise, places with the most search volume would always be ranked highest.
        To do this, each data point is divided by the total searches of the geography and time range it represents, to compare relative popularity.
        The resulting numbers are then scaled to a range of 0 to 100.
    """

    def __init__(self, path):

        self.datapath = 'data/'+path
        self.imgpath = 'img/'+path+"_json/"

        if not os.path.exists(self.imgpath):
            os.mkdir(self.imgpath)

        self.onlyfiles = [f for f in listdir(self.datapath) if isfile(join(self.datapath, f)) ]
        self.logfile = file(self.imgpath+"results-gtrend.txt", "w+")

        if not os.path.isfile(self.datapath+'/selected_people.csv'):
            print "Selected People File is missing! "
            raise Exception("Selected People File is missing!")


        self.people = pd.read_csv(self.datapath+'/selected_people.csv', delimiter=",", header=0, error_bad_lines=False)
        self.people = self.create_filename_col( self.people)

        self.people = self.people[~pd.isnull(self.people.birth_year) & (self.people.birth_year <= 2015)]
        self.people['birth_century'] = np.round(np.floor((self.people['birth_year']/100))*100)


    def create_filename_col(self, people):
         i = 0
         for ind,ser in people.iterrows():
            #print ind
            #print ser
            name = ser["label"]
            #print name
            if "(" in name:
                #remove additional info from name
                pos = name.find("(")
                name = name[0:pos]

            # remove the letter-dot stuff
            name = re.sub(r'\s[A-Z]\.\s', ' ', name)
            # remove quoted stuff e.g. Katharina "kate" Mcking
            name = re.sub(r'\s"[A-Za-z]+"\s', ' ', name)
            # remove stuff after the comma e.g. James Dean, King from Sweden
            name = re.sub(r',\s*.+', ' ', name)
            #print name
            people.ix[ind, "filename"] = name
            i = i+1
            if i % 10000 == 0:
                print i
         return people


    def correctDecember(selfself, string):
        items = string.split(',')
        if len(items) == 3:
            # month zero does not exist
            if (items[1] == '0'):
                month = int(items[0])
                items[0] = str(month - 1)
                items[1] = '12'

            items_str = items[0]+","+items[1]+","+items[2]
            return items_str
        else:
            return string


    def parse_gtrend_json(self, string):
        string = string.replace('// Data table response', '')
        string = string.replace('\n', '')
        string = string.replace('google.visualization.Query.setResponse(', '')
        string = string.replace('new Date', '')
        string = string[:-2]
        print string
        try:
            data = json.loads(string, encoding="utf8")
            print "could load json"
            print data
            return data
        except:
            searchObj = re.findall( r'"v":[0-9(),.]+', string, re.M|re.I)
            #searchObj = re.findall( r'"f":"[A-Za-z0-9\s]+"', string, re.M|re.I)
            if searchObj:
                #print "search --> searchObj.group() : ", searchObj
                items = []
                counter = 0
                skipFrequ = False
                for item in searchObj:
                    item = item.replace('"v":', '')
                    item = item.replace('"', '')
                    item = item.replace('(', '')
                    item = item.replace(')', '')
                    # remove last comma
                    item =  item[:-1]
                    #print (item)
                    item = self.correctDecember(item)
                    try:
                      #locale.setlocale(locale.LC_ALL, "deu_deu")
                        #print (item.encode("utf8"))
                        #items.append(datetime.datetime.strptime(item.encode("utf8"), "%B %Y")) #iso-8859-16
                        if ((counter % 2) == 0):
                            items.append(datetime.datetime.strptime(item.encode("utf8"), "%Y,%m,%d"))
                            skipFrequ = False

                        else:
                            if (not skipFrequ):
                                items.append(float(item))

                        counter += 1
                    except:
                        print "cannot parse %s"%item.encode("utf8")
                        skipFrequ = True
                        counter += 1
                #print items
                return items
            else:
                print "Nothing found!!"



    def run(self):
        regionalEntropy = {}
        regionCount = {}
        timeEntropy = {}
        sumInterest = {}
        posInterest = {}


        ##############################################################################
        # PARSE GOOGLE TREND RESULT FILES
        ##############################################################################
        quota_error = 0
        monthlyFiles = 0
        weeklyFiles = 0

        for file in self.onlyfiles:
            startFromLine = -1
            startFromLineTime = -1
            weekly = False
            pos = file.find(".json")
            if (pos >= 0):
                filename = file[0:pos]

                linesCounter = 1
                end = False
                endTime = False


                with open(self.datapath+"/"+file) as f:
                    content = f.read()
                    #print content
                    data = self.parse_gtrend_json(content)

                    timeseries = {}

                print data
                print len(data)
                if len(data) > 3:
                    for ind in xrange(0, len(data), 2):
                        #print data[ind]
                        #print data[ind+1]
                        if (ind+1) < len(data):
                            timeseries[data[ind]] = data[ind+1]


                timeFrequs = map(int, timeseries.values())

                if len(timeFrequs) > 1:
                    sumInterest[filename] = np.sum(timeFrequs)
                    posInterest[filename] = np.count_nonzero(timeFrequs)

                    if posInterest[filename] > 0:
                        timeEntropy[filename] = stats.entropy(timeFrequs)
                    else:
                        timeEntropy[filename] = np.nan


        interestDF = pd.DataFrame.from_dict(sumInterest.items())
        interestDF.columns=["filename", "timeInterest"]

        timeEntropyDF = pd.DataFrame.from_dict(timeEntropy.items())
        timeEntropyDF.columns=["filename", "timeEntropy"]

        posInterestDF = pd.DataFrame.from_dict(posInterest.items())
        posInterestDF.columns=["filename", "timePosInterest"]


        #print "regionalEntropyDF"
        #print regionalEntropyDF.head(n=1)
        #print regionalEntropyDF.shape
        #print "regionCountDF"
        #print regionCountDF.head(n=1)
        #print regionCountDF.shape
        #print "self.people"
        #print self.people.head(n=1)
        #print self.people.shape

        # add the computed statistics to the people file
        self.people = self.people.merge(timeEntropyDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(interestDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(posInterestDF, right_on="filename", left_on="filename", how="inner")


        print "AFTER MERGING"
        print self.people.head(n=1)
        print self.people.shape





        men = self.people[self.people.gender =="male"]
        women = self.people[self.people.gender =="female"]

        labels = ['female ('+str(len(women.index))+')', 'male ('+str(len(men.index))+')']

        ##############################################################################
        # PLOTS TOTAL INTEREST
        ##############################################################################

        data = [women.timeInterest.values, men.timeInterest.values]
        self.logfile.write("\n \n Mann Withney U Temp Sum Interest:")
        U, p =  stats.mstats.mannwhitneyu(women.timeInterest.values, men.timeInterest.values)
        ut.write_mannwithneyu(U, p, women.timeInterest.values, men.timeInterest.values, self.logfile)
        self.make_boxplot(data, labels, self.imgpath+"gtrend_time_interest_box.png", "sum interest")
        self.plot_ccdf(np.array(women.timeInterest.values.tolist()), np.array(men.timeInterest.values.tolist()), labels, self.imgpath+"gtrend_time_interest_ccdf.png", "Sum Interest", 1, 0)
        ut.plot_facet_dist(self.people, 'gender', 'timeInterest', self.imgpath+"gtrend_time_interest.png")

        data = [women.timePosInterest.values, men.timePosInterest.values]
        self.logfile.write("\n\n Mann Withney U Temp Pos Interest:")
        U, p =  stats.mstats.mannwhitneyu(women.timePosInterest.values, men.timePosInterest.values)
        ut.write_mannwithneyu(U, p, women.timePosInterest.values, men.timePosInterest.values, self.logfile)
        self.make_boxplot(data, labels, self.imgpath+"gtrend_time_pos_interest_box.png", "num weeks with interest")
        self.plot_ccdf(np.array(women.timePosInterest.values.tolist()), np.array(men.timePosInterest.values.tolist()), labels, self.imgpath+"gtrend_time_pos_interest_ccdf.png", "Num weeks with interest", 1, 0)
        ut.plot_facet_dist(self.people, 'gender', 'timePosInterest', self.imgpath+"gtrend_time_pos_interest.png")


        ##############################################################################
        # PLOT Entropy Temp INTEREST
        ##############################################################################
        limPeople = self.people[np.isfinite(self.people['timeEntropy'])] #people[people.index not in inds]
        men = limPeople[limPeople.gender =="male"]
        women = limPeople[limPeople.gender =="female"]
        data = [women.timeEntropy.values, men.timeEntropy.values]
        self.logfile.write("\n\n Mann Withney U Time Entropy:")
        U, p =  stats.mstats.mannwhitneyu(women.timeEntropy.values, men.timeEntropy.values)
        ut.write_mannwithneyu(U, p, women.timeEntropy.values, men.timeEntropy.values, self.logfile)
        self.make_boxplot(data, labels, self.imgpath+"gtrend_time_entropy_box.png", "temporal entropy")
        self.plot_ccdf(np.array(women.timeEntropy.values.tolist()), np.array(men.timeEntropy.values.tolist()), labels, self.imgpath+"gtrend_time_entropy_ccdf.png", "Temp Entropy", 1, 0)
        ut.plot_facet_dist(self.people, 'gender', 'timeEntropy', self.imgpath+"gtrend_time_entropy.png")



        self.regression()




    def regression(self):

        print self.people.head(n=1)
        self.people.rename(columns={'class': 'dbpedia_class'}, inplace=True) # all_bios is the dataframe with the consolidated data. somehow it doesn't work if the class column is named "class"


        self.logfile.write( "\n\n Sum Temp Interest NegBinom")
        m = glm("timeInterest ~ C(gender,Treatment(reference='male')) ", data=self.people, family=families.NegativeBinomial()).fit()
        self.logfile.write( "\n AIC"+str(+m.aic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())

        self.logfile.write( "\n\n Sum Temp Interest OLS")
        m = ols("timeInterest ~ C(gender,Treatment(reference='male')) ", data=self.people).fit()
        self.logfile.write( "\n AIC"+str(+m.aic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())


        self.logfile.write( "\n\n Pos Temp Interest NegBinom")
        m = glm("timePosInterest ~ C(gender,Treatment(reference='male')) ", data=self.people, family=families.NegativeBinomial()).fit()
        self.logfile.write( "\n AIC "+str(m.aic))
        self.logfile.write( "\n BIC "+str(m.bic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())

        #lim_people = self.people[self.people.timePosInterest>0]
        self.logfile.write( "\n\n Pos Temp Interest OLS")
        m = ols("timePosInterest ~ C(gender,Treatment(reference='male')) ", data=self.people).fit()
        self.logfile.write( "\n AIC "+str(m.aic))
        self.logfile.write( "\n BIC "+str(m.bic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())


        # Beta regression for normalized entropy could work
        #print "\n\n Time Entropy"
        #m = ols("timeEntropy ~ C(gender,Treatment(reference='male')) ", #+ C(dbpedia_class,Treatment(reference='http://dbpedia.org/ontology/Person')) + birth_century
        #        data=self.people).fit()
        #print m.summary() # <-- this gives you the table of coefficients with p-values, confidence intervals, and so on


    def make_boxplot(self, data, labels, filename, ylabel):
        plt.figure()
        plt.boxplot(data)
        # mark the mean
        means = [np.mean(x) for x in data]
        print ylabel
        print means
        #print range(1, len(data)+1)
        plt.scatter(range(1, len(data)+1), means, color="red", marker=">", s=20)
        plt.ylabel(ylabel)
        plt.xticks(range(1, len(data)+1), labels)
        plt.savefig(filename)

        #plt.figure()
        #plt.boxplot(data)
        ## mark the mean
        #means = [np.mean(x) for x in data]
        #print "entropy means"
        #print means
        #print range(1, len(data)+1)
        #plt.scatter(range(1, len(data)+1), means, color="red", marker=">", s=20)
        #plt.ylabel('shannon entropy')
        #plt.xticks(range(1, len(data)+1), labels)
        #plt.savefig('./img/gtrend_region_entropy.png')


    def plot_ccdf(self, women_values, men_values, labels, filename, xlabel, xlog, ylog):
        item_frequency_female = itemfreq(women_values)
        item_frequency_male = itemfreq(men_values)
        ccdf= 1
        ut.plot_cdf(list([item_frequency_female, item_frequency_male]), labels, ['pink','blue'], filename, xlabel, ccdf, xlog, ylog)

        #item_frequency_female = itemfreq(np.array(women.entropy.values.tolist()))
        #item_frequency_male = itemfreq(np.array(men.entropy.values.tolist()))
        #ccdf = 1
        #ut.plotCDF(list([item_frequency_female, item_frequency_male]), labels, ['pink','blue'], './img/gtrend_region_entropy_ccdf.png', 'Entropy', ccdf, 0, 0)




if __name__ == "__main__":

    #analyzer = GoogleTrendAnalyzer('trends')
    #analyzer.run()

    #analyzer = GoogleTrendAnalyzer('trends-sample-birth1946')
    #analyzer.run()

    #test = '{"version":"0.6","status":"ok","sig":"1652532895","table":{"cols":[{"id":"date","label":"Date","type":"date","pattern":""},{"id":"query0","label":"adam wood","type":"number","pattern":""}],' \
    #       '"rows":[{"c":[{"v":new Date(2004,0,1),"f":"Dezember 2003"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,1,1),"f":"Januar 2004"},{"v":0.0,"f":"0"}]},' \
    #       '{"c":[{"v":new Date(2004,2,1),"f":"Februar 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,3,1),"f":"März 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,4,1),"f":"April 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,5,1),"f":"Mai 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,6,1),"f":"Juni 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,7,1),"f":"Juli 2004"},{"v":0.0,"f":"0"}]},{"c":[{"v":new Date(2004,8,1),"f":"August 2004"},{"v":97.0,"f":"97"}]},{"c":[{"v":new Date(2004,9,1),"f":"September 2004"},{"v":80.0,"f":"80"}]},{"c":[{"v":new Date(2004,10,1),"f":"Oktober 2004"},{"v":80.0,"f":"80"}]},{"c":[{"v":new Date(2004,11,1),"f":"November 2004"},{"v":80.0,"f":"80"}]},{"c":[{"v":new Date(2005,0,1),"f":"Dezember 2004"},{"v":86.0,"f":"86"}]},{"c":[{"v":new Date(2005,1,1),"f":"Januar 2005"},{"v":91.0,"f":"91"}]},{"c":[{"v":new Date(2005,2,1),"f":"Februar 2005"},{"v":97.0,"f":"97"}]},{"c":[{"v":new Date(2005,3,1),"f":"März 2005"},{"v":67.0,"f":"67"}]},{"c":[{"v":new Date(2005,4,1),"f":"April 2005"},{"v":67.0,"f":"67"}]},{"c":[{"v":new Date(2005,5,1),"f":"Mai 2005"},{"v":67.0,"f":"67"}]},{"c":[{"v":new Date(2005,6,1),"f":"Juni 2005"},{"v":62.0,"f":"62"}]},{"c":[{"v":new Date(2005,7,1),"f":"Juli 2005"},{"v":79.0,"f":"79"}]},{"c":[{"v":new Date(2005,8,1),"f":"August 2005"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2005,9,1),"f":"September 2005"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2005,10,1),"f":"Oktober 2005"},{"v":51.0,"f":"51"}]},{"c":[{"v":new Date(2005,11,1),"f":"November 2005"},{"v":62.0,"f":"62"}]},{"c":[{"v":new Date(2006,0,1),"f":"Dezember 2005"},{"v":65.0,"f":"65"}]},{"c":[{"v":new Date(2006,1,1),"f":"Januar 2006"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2006,2,1),"f":"Februar 2006"},{"v":63.0,"f":"63"}]},{"c":[{"v":new Date(2006,3,1),"f":"März 2006"},{"v":71.0,"f":"71"}]},{"c":[{"v":new Date(2006,4,1),"f":"April 2006"},{"v":57.0,"f":"57"}]},{"c":[{"v":new Date(2006,5,1),"f":"Mai 2006"},{"v":50.0,"f":"50"}]},{"c":[{"v":new Date(2006,6,1),"f":"Juni 2006"},{"v":43.0,"f":"43"}]},{"c":[{"v":new Date(2006,7,1),"f":"Juli 2006"},{"v":66.0,"f":"66"}]},{"c":[{"v":new Date(2006,8,1),"f":"August 2006"},{"v":44.0,"f":"44"}]},{"c":[{"v":new Date(2006,9,1),"f":"September 2006"},{"v":32.0,"f":"32"}]},{"c":[{"v":new Date(2006,10,1),"f":"Oktober 2006"},{"v":68.0,"f":"68"}]},{"c":[{"v":new Date(2006,11,1),"f":"November 2006"},{"v":36.0,"f":"36"}]},{"c":[{"v":new Date(2007,0,1),"f":"Dezember 2006"},{"v":47.0,"f":"47"}]},{"c":[{"v":new Date(2007,1,1),"f":"Januar 2007"},{"v":41.0,"f":"41"}]},{"c":[{"v":new Date(2007,2,1),"f":"Februar 2007"},{"v":39.0,"f":"39"}]},{"c":[{"v":new Date(2007,3,1),"f":"März 2007"},{"v":44.0,"f":"44"}]},{"c":[{"v":new Date(2007,4,1),"f":"April 2007"},{"v":49.0,"f":"49"}]},{"c":[{"v":new Date(2007,5,1),"f":"Mai 2007"},{"v":41.0,"f":"41"}]},{"c":[{"v":new Date(2007,6,1),"f":"Juni 2007"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2007,7,1),"f":"Juli 2007"},{"v":53.0,"f":"53"}]},{"c":[{"v":new Date(2007,8,1),"f":"August 2007"},{"v":44.0,"f":"44"}]},{"c":[{"v":new Date(2007,9,1),"f":"September 2007"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2007,10,1),"f":"Oktober 2007"},{"v":34.0,"f":"34"}]},{"c":[{"v":new Date(2007,11,1),"f":"November 2007"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2008,0,1),"f":"Dezember 2007"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2008,1,1),"f":"Januar 2008"},{"v":54.0,"f":"54"}]},{"c":[{"v":new Date(2008,2,1),"f":"Februar 2008"},{"v":65.0,"f":"65"}]},{"c":[{"v":new Date(2008,3,1),"f":"März 2008"},{"v":41.0,"f":"41"}]},{"c":[{"v":new Date(2008,4,1),"f":"April 2008"},{"v":57.0,"f":"57"}]},{"c":[{"v":new Date(2008,5,1),"f":"Mai 2008"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2008,6,1),"f":"Juni 2008"},{"v":43.0,"f":"43"}]},{"c":[{"v":new Date(2008,7,1),"f":"Juli 2008"},{"v":42.0,"f":"42"}]},{"c":[{"v":new Date(2008,8,1),"f":"August 2008"},{"v":45.0,"f":"45"}]},{"c":[{"v":new Date(2008,9,1),"f":"September 2008"},{"v":56.0,"f":"56"}]},{"c":[{"v":new Date(2008,10,1),"f":"Oktober 2008"},{"v":43.0,"f":"43"}]},{"c":[{"v":new Date(2008,11,1),"f":"November 2008"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2009,0,1),"f":"Dezember 2008"},{"v":68.0,"f":"68"}]},{"c":[{"v":new Date(2009,1,1),"f":"Januar 2009"},{"v":50.0,"f":"50"}]},{"c":[{"v":new Date(2009,2,1),"f":"Februar 2009"},{"v":56.0,"f":"56"}]},{"c":[{"v":new Date(2009,3,1),"f":"März 2009"},{"v":42.0,"f":"42"}]},{"c":[{"v":new Date(2009,4,1),"f":"April 2009"},{"v":58.0,"f":"58"}]},{"c":[{"v":new Date(2009,5,1),"f":"Mai 2009"},{"v":54.0,"f":"54"}]},{"c":[{"v":new Date(2009,6,1),"f":"Juni 2009"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2009,7,1),"f":"Juli 2009"},{"v":97.0,"f":"97"}]},{"c":[{"v":new Date(2009,8,1),"f":"August 2009"},{"v":80.0,"f":"80"}]},{"c":[{"v":new Date(2009,9,1),"f":"September 2009"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2009,10,1),"f":"Oktober 2009"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2009,11,1),"f":"November 2009"},{"v":58.0,"f":"58"}]},{"c":[{"v":new Date(2010,0,1),"f":"Dezember 2009"},{"v":53.0,"f":"53"}]},{"c":[{"v":new Date(2010,1,1),"f":"Januar 2010"},{"v":48.0,"f":"48"}]},{"c":[{"v":new Date(2010,2,1),"f":"Februar 2010"},{"v":54.0,"f":"54"}]},{"c":[{"v":new Date(2010,3,1),"f":"März 2010"},{"v":73.0,"f":"73"}]},{"c":[{"v":new Date(2010,4,1),"f":"April 2010"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2010,5,1),"f":"Mai 2010"},{"v":52.0,"f":"52"}]},{"c":[{"v":new Date(2010,6,1),"f":"Juni 2010"},{"v":64.0,"f":"64"}]},{"c":[{"v":new Date(2010,7,1),"f":"Juli 2010"},{"v":67.0,"f":"67"}]},{"c":[{"v":new Date(2010,8,1),"f":"August 2010"},{"v":59.0,"f":"59"}]},{"c":[{"v":new Date(2010,9,1),"f":"September 2010"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2010,10,1),"f":"Oktober 2010"},{"v":82.0,"f":"82"}]},{"c":[{"v":new Date(2010,11,1),"f":"November 2010"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2011,0,1),"f":"Dezember 2010"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2011,1,1),"f":"Januar 2011"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2011,2,1),"f":"Februar 2011"},{"v":76.0,"f":"76"}]},{"c":[{"v":new Date(2011,3,1),"f":"März 2011"},{"v":47.0,"f":"47"}]},{"c":[{"v":new Date(2011,4,1),"f":"April 2011"},{"v":48.0,"f":"48"}]},{"c":[{"v":new Date(2011,5,1),"f":"Mai 2011"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2011,6,1),"f":"Juni 2011"},{"v":51.0,"f":"51"}]},{"c":[{"v":new Date(2011,7,1),"f":"Juli 2011"},{"v":72.0,"f":"72"}]},{"c":[{"v":new Date(2011,8,1),"f":"August 2011"},{"v":68.0,"f":"68"}]},{"c":[{"v":new Date(2011,9,1),"f":"September 2011"},{"v":57.0,"f":"57"}]},{"c":[{"v":new Date(2011,10,1),"f":"Oktober 2011"},{"v":62.0,"f":"62"}]},{"c":[{"v":new Date(2011,11,1),"f":"November 2011"},{"v":43.0,"f":"43"}]},{"c":[{"v":new Date(2012,0,1),"f":"Dezember 2011"},{"v":57.0,"f":"57"}]},{"c":[{"v":new Date(2012,1,1),"f":"Januar 2012"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2012,2,1),"f":"Februar 2012"},{"v":50.0,"f":"50"}]},{"c":[{"v":new Date(2012,3,1),"f":"März 2012"},{"v":62.0,"f":"62"}]},{"c":[{"v":new Date(2012,4,1),"f":"April 2012"},{"v":68.0,"f":"68"}]},{"c":[{"v":new Date(2012,5,1),"f":"Mai 2012"},{"v":49.0,"f":"49"}]},{"c":[{"v":new Date(2012,6,1),"f":"Juni 2012"},{"v":51.0,"f":"51"}]},{"c":[{"v":new Date(2012,7,1),"f":"Juli 2012"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2012,8,1),"f":"August 2012"},{"v":68.0,"f":"68"}]},{"c":[{"v":new Date(2012,9,1),"f":"September 2012"},{"v":44.0,"f":"44"}]},{"c":[{"v":new Date(2012,10,1),"f":"Oktober 2012"},{"v":54.0,"f":"54"}]},{"c":[{"v":new Date(2012,11,1),"f":"November 2012"},{"v":69.0,"f":"69"}]},{"c":[{"v":new Date(2013,0,1),"f":"Dezember 2012"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2013,1,1),"f":"Januar 2013"},{"v":66.0,"f":"66"}]},{"c":[{"v":new Date(2013,2,1),"f":"Februar 2013"},{"v":66.0,"f":"66"}]},{"c":[{"v":new Date(2013,3,1),"f":"März 2013"},{"v":56.0,"f":"56"}]},{"c":[{"v":new Date(2013,4,1),"f":"April 2013"},{"v":53.0,"f":"53"}]},{"c":[{"v":new Date(2013,5,1),"f":"Mai 2013"},{"v":46.0,"f":"46"}]},{"c":[{"v":new Date(2013,6,1),"f":"Juni 2013"},{"v":65.0,"f":"65"}]},{"c":[{"v":new Date(2013,7,1),"f":"Juli 2013"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2013,8,1),"f":"August 2013"},{"v":79.0,"f":"79"}]},{"c":[{"v":new Date(2013,9,1),"f":"September 2013"},{"v":59.0,"f":"59"}]},{"c":[{"v":new Date(2013,10,1),"f":"Oktober 2013"},{"v":100.0,"f":"100"}]},{"c":[{"v":new Date(2013,11,1),"f":"November 2013"},{"v":64.0,"f":"64"}]},{"c":[{"v":new Date(2014,0,1),"f":"Dezember 2013"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2014,1,1),"f":"Januar 2014"},{"v":50.0,"f":"50"}]},{"c":[{"v":new Date(2014,2,1),"f":"Februar 2014"},{"v":56.0,"f":"56"}]},{"c":[{"v":new Date(2014,3,1),"f":"März 2014"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2014,4,1),"f":"April 2014"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2014,5,1),"f":"Mai 2014"},{"v":65.0,"f":"65"}]},{"c":[{"v":new Date(2014,6,1),"f":"Juni 2014"},{"v":70.0,"f":"70"}]},{"c":[{"v":new Date(2014,7,1),"f":"Juli 2014"},{"v":63.0,"f":"63"}]},{"c":[{"v":new Date(2014,8,1),"f":"August 2014"},{"v":45.0,"f":"45"}]},{"c":[{"v":new Date(2014,9,1),"f":"September 2014"},{"v":69.0,"f":"69"}]},{"c":[{"v":new Date(2014,10,1),"f":"Oktober 2014"},{"v":58.0,"f":"58"}]},{"c":[{"v":new Date(2014,11,1),"f":"November 2014"},{"v":53.0,"f":"53"}]},{"c":[{"v":new Date(2015,0,1),"f":"Dezember 2014"},{"v":57.0,"f":"57"}]},{"c":[{"v":new Date(2015,1,1),"f":"Januar 2015"},{"v":69.0,"f":"69"}]},{"c":[{"v":new Date(2015,2,1),"f":"Februar 2015"},{"v":61.0,"f":"61"}]},{"c":[{"v":new Date(2015,3,1),"f":"März 2015"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2015,4,1),"f":"April 2015"},{"v":65.0,"f":"65"}]},{"c":[{"v":new Date(2015,5,1),"f":"Mai 2015"},{"v":58.0,"f":"58"}]},{"c":[{"v":new Date(2015,6,1),"f":"Juni 2015"},{"v":60.0,"f":"60"}]},{"c":[{"v":new Date(2015,7,1),"f":"Juli 2015"},{"v":63.0,"f":"63"}]},{"c":[{"v":new Date(2015,8,1),"f":"August 2015"},{"v":51.0,"f":"51"}]},{"c":[{"v":new Date(2015,9,1),"f":"September 2015"},{"v":55.0,"f":"55"}]},{"c":[{"v":new Date(2015,10,1),"f":"Oktober 2015"},{"v":50.0,"f":"50"}]}]}}';
    #test =  test.replace('new Date', '')
     #test = test.replace('google.visualization.Query.setResponse(', '')
    #test = test[:-2]
    #print test


    analyzer = GoogleTrendAnalyzer('trends-sample-birth1900')
    analyzer.run()