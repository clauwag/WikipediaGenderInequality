__author__ = 'wagnerca'

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


class GoogleTrendAnalyzer:
    """
        Analyze Google Trend Output
        Google Trends adjusts search data to make comparisons between terms easier. Otherwise, places with the most search volume would always be ranked highest.
        To do this, each data point is divided by the total searches of the geography and time range it represents, to compare relative popularity.
        The resulting numbers are then scaled to a range of 0 to 100.
    """

    def __init__(self, path):

        self.datapath = 'data/'+path
        self.imgpath = 'img/'+path+"/"

        if not os.path.exists(self.imgpath):
            os.mkdir(self.imgpath)

        self.onlyfiles = [f for f in listdir(self.datapath) if isfile(join(self.datapath, f)) ]
        self.logfile = file(self.imgpath+"results-gtrend.txt", "w+")

        if not os.path.isfile(self.datapath+'/selected_people.csv'):
            print "Selected People File is missing! "
            raise Exception("Selected People File is missing!")
            # We could also create the file when it is missing
            #self.allpeople = pd.read_csv('data/person_data.csv', delimiter=",", header=0)
            #print self.allpeople.shape
            #self.allpeople = self.create_filename_col(self.allpeople)
            #self.create_selected_people_file()

        self.people = pd.read_csv(self.datapath+'/selected_people.csv', delimiter=",", header=0, error_bad_lines=False)
        self.people = self.create_filename_col( self.people)

        #people_with_birthyear = self.people[self.people["birth_year"] > 0]
        self.people = self.people[~pd.isnull(self.people.birth_year) & (self.people.birth_year <= 2015)]

        self.people['birth_century'] = np.round(np.floor((self.people['birth_year']/100))*100)



    def create_selected_people_file(self):
        filenames=[]
        for file in self.onlyfiles:
            #print file
            pos = file.find(".csv")
            filenames.append(file[0:pos])

        #print len(filenames)
        selected_filenames = pd.DataFrame({"filename":filenames})

        print selected_filenames.head(n=1)
        print selected_filenames.shape

        # add the computed statistics to the people file
        selected_people = self.allpeople.merge(selected_filenames, right_on="filename", left_on="filename", how="inner")

        print "AFTER MERGING selected_people"
        print selected_people.head(n=1)
        print selected_people.shape

        selected_people.to_csv(self.datapath+"/selected_people.csv")



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
        for file in self.onlyfiles:
            startFromLine = -1
            startFromLineTime = -1

            pos = file.find(".csv")
            filename = file[0:pos]

            linesCounter = 1
            end = False
            endTime = False

            with open(self.datapath+"/"+file) as f:
                content = f.readlines()
                regions = {}
                timeseries = {}
                for line in content:
                    if line.startswith("<div id="):
                        quota_error += 1
                        print "quota error for %s"%filename
                        break;
                    if line.startswith("Region,"):
                        startFromLine = linesCounter
                    if line.startswith("Month,"):
                        startFromLineTime = linesCounter

                    if ((startFromLine > 0) and (linesCounter > startFromLine) and (not end)):
                         if line == "\n":
                            end = True
                         else:
                            items  = line.split(",")
                            regions[items[0]] = items[1]

                    if ((startFromLineTime > 0) and (linesCounter > startFromLineTime) and (not endTime)):
                        print line
                        if line == "\n":
                            endTime = True

                        else:
                            items  = line.split(",")
                            if items[1] == ' \n': # sometimes gtrends returns empty field rather than 0
                                timeseries[items[0]] = "0"
                            else:
                                timeseries[items[0]] = items[1]
                    linesCounter += 1


            timeFrequs = map(int, timeseries.values())
            regionFrequs = map(int,(regions.values()))

            if linesCounter > 1:
                sumInterest[filename] = np.sum(timeFrequs)
                posInterest[filename] = np.count_nonzero(timeFrequs)

                if posInterest[filename] > 0:
                    timeEntropy[filename] = stats.entropy(timeFrequs)
                else:
                    timeEntropy[filename] = np.nan

                regionCount[filename] = len(regionFrequs)
                if(np.sum(regionFrequs) > 0):
                    regionalEntropy[filename] = stats.entropy(regionFrequs)
                else:
                    regionalEntropy[filename] = np.nan


        # store results into a dataframe
        regionalEntropyDF = pd.DataFrame.from_dict(regionalEntropy.items())
        regionalEntropyDF.columns=["filename", "entropy"]
        print regionalEntropyDF.head()

        regionCountDF = pd.DataFrame.from_dict(regionCount.items())
        regionCountDF.columns=["filename", "numRegions"]

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
        self.people = self.people.merge(regionalEntropyDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(regionCountDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(timeEntropyDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(interestDF, right_on="filename", left_on="filename", how="inner")
        self.people = self.people.merge(posInterestDF, right_on="filename", left_on="filename", how="inner")


        print "AFTER MERGING"
        print self.people.head(n=1)
        print self.people.shape




        ##############################################################################
        # PLOTS NUM REGIONS
        ##############################################################################

        men = self.people[self.people.gender =="male"]
        women = self.people[self.people.gender =="female"]

        labels = ['female ('+str(len(women.index))+')', 'male ('+str(len(men.index))+')']
        data = [women.numRegions.values, men.numRegions.values]

        self.logfile.write("\n Mann Withney U Num regions:")
        U, p =  stats.mstats.mannwhitneyu(women.numRegions.values, men.numRegions.values)
        ut.write_mannwithneyu(U, p, women.numRegions.values, men.numRegions.values, self.logfile)
        self.make_boxplot(data, labels, self.imgpath+"gtrend_num_regions_box.png", "num regions")
        self.plot_ccdf(np.array(women.numRegions.values.tolist()), np.array(men.numRegions.values.tolist()), labels, self.imgpath+"gtrend_num_regions_ccdf.png", "Num Regions", 1, 0)
        ut.plot_facet_dist(self.people, 'gender', 'numRegions', self.imgpath+"gtrend_num_regions.png")
        ut.rank_size_plot(self.people, 'numRegions', 'Num Regions Gtrends',  self.imgpath+"gtrend_num_regions_ranksize.png")

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


        ##############################################################################
        # PLOT ENTROPY
        ##############################################################################
        # for entropy we need to remove the nan value. If we dont have data the result is nan
        limPeople = self.people[np.isfinite(self.people['entropy'])] #people[people.index not in inds]
        men = limPeople[limPeople.gender =="male"]
        women = limPeople[limPeople.gender =="female"]
        labels = ['female ('+str(len(women.index))+')', 'male ('+str(len(men.index))+')']
        data = [women.entropy.values, men.entropy.values]

        self.logfile.write("\n\n Mann Withney U Entropy:")
        U, p =  stats.mstats.mannwhitneyu(women.entropy.values, men.entropy.values)
        ut.write_mannwithneyu(U, p, women.entropy.values, men.entropy.values, self.logfile)
        self.make_boxplot(data, labels, "gtrend_region_entropy_box.png", "shannon entropy")
        self.plot_ccdf(np.array(women.entropy.values.tolist()), np.array(men.entropy.values.tolist()), labels, self.imgpath+"gtrend_entropy_ccdf.png", "Entropy", 0, 0)
        ut.plot_facet_dist(self.people, 'gender', 'entropy', self.imgpath+"gtrend_region_entropy.png")


        self.regression()




    def regression(self):

        print self.people.head(n=1)
        self.people.rename(columns={'class': 'dbpedia_class'}, inplace=True) # all_bios is the dataframe with the consolidated data. somehow it doesn't work if the class column is named "class"

        self.logfile.write( "\n\n Num Regions NegativeBinomial")
        m = glm("numRegions ~ C(gender,Treatment(reference='male')) ", # + C(dbpedia_class,Treatment(reference='http://dbpedia.org/ontology/Person')) + birth_century
                data=self.people, family=families.NegativeBinomial()).fit()
        self.logfile.write( "\n AIC "+str(m.aic))
        self.logfile.write( "\n BIC "+str(m.bic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())


        #lim_people = self.people[self.people.numRegions>0]
        self.logfile.write( "\n\n Num Regions OLS")
        m = ols("numRegions ~ C(gender,Treatment(reference='male')) ", # + C(dbpedia_class,Treatment(reference='http://dbpedia.org/ontology/Person')) + birth_century
                data=self.people).fit()
        self.logfile.write( "\n AIC "+str(m.aic))
        self.logfile.write( "\n BIC "+str(m.bic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())



        # we could use beta regression for normalized entropy
        #print "\n\n Region Entropy"
        #m = ols("entropy ~ C(gender,Treatment(reference='male')) ", #+ C(dbpedia_class,Treatment(reference='http://dbpedia.org/ontology/Person')) + birth_century
        #        data=self.people).fit()
        #print m.summary() # <-- this gives you the table of coefficients with p-values, confidence intervals, and so on



        self.logfile.write( "\n\n Sum Temp Interest")
        m = ols("timeInterest ~ C(gender,Treatment(reference='male')) ", data=self.people).fit()
        self.logfile.write( "\n AIC"+str(+m.aic))
        for table in m.summary().tables:
            self.logfile.write(table.as_latex_tabular())


        self.logfile.write( "\n\n Pos Temp Interest")
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

    analyzer = GoogleTrendAnalyzer('trends-sample-birth1900')
    analyzer.run()