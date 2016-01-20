from __future__ import print_function

__author__ = 'claudia wagner'

import pandas as pd
import numpy as np
from scipy.stats import itemfreq
import scipy.stats as stats
import util as ut
import pylab as plt
import os
import statsmodels.api as sm

class GlobalWikipediaPopularity:
    """
    This class compares the global notebility men and women in Wikipedia using the number of language editions
    in which an article exists as a proxy for notability
    """

    def __init__(self, datapath, start, end):
        self.people = pd.read_csv(datapath, delimiter=",", header=0)

        print ("total num people %s"%len(self.people.index))
        print (" num people not in EN %s"%len(self.people[(self.people.available_english == False)].index))


        self.people = self.people[~pd.isnull(self.people.birth_year) & (self.people.birth_year <= 2015)]

        # Create 'decade' and 'century' column
        self.people['birth_decade'] = np.round(np.floor((self.people['birth_year']/10))*10)
        self.people['birth_century'] = np.round(np.floor((self.people['birth_year']/100))*100)

        self.pre = ""
        self.post = ""
        if start >= 0 and end >= 0:
            self.pre = str(start)
            self.post = str(end)
            recent_people = self.people[(self.people.birth_year > start) & (self.people.birth_year <= end)]
            self.people = recent_people
            print ("num people born between %s and %s century %s "%(start, end, len(self.people.index)))
            print ("num people born between %s and %s century %s "%(start, end, len(recent_people.index)))
            print ("num people for which gender is available AND they are born between %s and 2015: %s "
			%(year, len( recent_people[(recent_people.gender == "male") | (recent_people.gender == "female")])))


        self.logfile = file("img/results-numlang"+self.pre+"-"+self.post+".txt", "w+")




    def genderBoxplots(self, women, men, labels, path):
        data = [women.edition_count.values, men.edition_count.values]

        plt.figure()
        plt.boxplot(data)

        # mark the mean
        means = [np.mean(x) for x in data]
        print (means)

        plt.scatter(range(1, len(data)+1), means, color="red", marker=">", s=20)
        plt.ylabel('num editions')
        plt.xticks(range(1, len(data)+1), labels)
        plt.savefig(path+'/numeditions_gender_box_withOutlier'+self.pre+"-"+self.post+'.png', bbox_inches="tight")

        plt.figure()
        plt.boxplot(data, sym='')
        # mark the mean
        means = [np.mean(x) for x in data]
        print (means)

        plt.scatter(range(1, len(data)+1), means, color="red", marker=">", s=20)
        plt.ylabel('num editions')
        plt.xticks(range(1, len(data)+1), labels)
        plt.savefig(path+'/numeditions_gender_box'+self.pre+"-"+self.post+'.png', bbox_inches="tight")




    def analyze_numlangedition_per_decade(self):
        interval = 10
        decade = 1000
        #decades_of_interest = np.arange(0, 2010, interval)

        res = pd.DataFrame(columns = ('class', 'mean-men', "sem-men", 'mean-women', 'sem-women'))
        #for decade in decades_of_interest:
        print (self.people.shape)
        print (self.people.head(n=1))

        #print (self.people[(self.people.birth_year < 1000) & (self.people.birth_year > 0)].shape)
        people = self.people[(self.people.birth_year < decade) & (self.people.birth_year >= 0 )]
        print (people.shape)
        resdict = self.analyze_num_lang_edition("0-1000", people)
        res = res.append(resdict, ignore_index = True)

        while (decade < 2015):

            end  = decade+interval

            people = self.people[(self.people.birth_year >= decade) & (self.people.birth_year < end)]
            print (decade)
            print (decade+interval)
            print (people.shape)
            resdict = self.analyze_num_lang_edition(str(decade)+"-"+str(end), people)
            res = res.append(resdict, ignore_index = True)
            decade = end

     
        ut.plot_shaded_lines(res["class"].values, res["mean-women"].values, res["mean-men"].values, res["sem-women"].values, res["sem-women"].values, 'Mean Num Editions', 'Birth Year',
                             'img/numedition_gender_deacdes'+self.pre+"-"+self.post+'.png')




    def analyze_numlangedition_per_profession(self):
        classes = self.people["class"].unique()
        print (self.people["class"].nunique())
        #print classes

        res = pd.DataFrame(columns = ('class', 'mean-men', "sem-men", 'mean-women', 'sem-women'))
        for pclass in classes:

            people = self.people[self.people["class"] == pclass]

            print (pclass)
            classname = pclass.split("/")[-1]
            print (classname)
            resdict = self.analyze_num_lang_edition(classname, people)
            res = res.append(resdict, ignore_index = True)

        ut.plot_shaded_lines(res["class"].values, res["mean-women"].values, res["mean-men"].values, res["sem-women"].values, res["sem-women"].values, 'mean num editions', 'professions', 'img/numedition_gender_deacdes.png')


    def get_ratio(self, female, male, normalize):
        female_frequency_sorted = female[:, 1]
        female_numedition_sorted = female[:,0]
        male_frequency_sorted = male[:, 1]
        male_numedition_sorted = male[:,0]

        female_max = np.max(female_numedition_sorted)
        male_max = np.max(male_numedition_sorted)
        max = np.max([female_max, male_max])
        male_sum = np.sum(male_frequency_sorted.item(ind) for ind in range(0, len(male_frequency_sorted)))
        female_sum = np.sum(female_frequency_sorted.item(ind) for ind in range(0, len(female_frequency_sorted)))
        ratio = {}
        for i in range(1, max):
            ind = np.where(male_numedition_sorted == i)
            #print (ind)

            if (ind > 0 and len(male_frequency_sorted[ind] == 1)):
                if normalize:
                    male_val = male_frequency_sorted.item(ind) / float(male_sum)
                else:
                    male_val = male_frequency_sorted.item(ind)
            else:
                male_val = 0

            ind_f = np.where(female_numedition_sorted == i)
            if(ind_f > 0 and len(female_frequency_sorted[ind_f]) == 1):
                if normalize:
                    female_val = female_frequency_sorted.item(ind_f) / float(female_sum)
                else:
                    female_val = female_frequency_sorted.item(ind_f)

                ratio[i] = male_val/float(female_val)
            #else:
            #    ratio.append(male_val/female_val)
        return ratio



    def analyze_num_lang_edition(self, classname, people):
        path = "img/"+classname

        men = people[people.gender =="male"]
        women = people[people.gender =="female"]


        #create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        self.logfile.write("\n\n\n\ "+classname+"\n")
        edition_counts = np.append(women.edition_count.values, men.edition_count.values)
        labels = ['female ('+str(len(women.index))+')', 'male ('+str(len(men.index))+')']

        #print edition_counts
        #print type(edition_counts)probability that a score randomly drawn from population A will be greater than a score randomly drawn from population B.
        max_num_editions = np.max(edition_counts)
        #print max_num_editions
        top_men = []
        men_percentage = 0
        topk = range(10, 110, 10)

        if (men.shape[0] > 0):
            men_percentage = [men[men.edition_count == x].shape[0]/float(men.shape[0]) for x in range(1, max_num_editions)]
            self.logfile.write("\n % of local men ")
            self.logfile.write(str(men[men.edition_count <2].shape[0]/float(men.shape[0])))
            self.logfile.write("\n percentage men %s"%str(len(men.index)/float((len(women.index)+len(men.index)))))
            men_vals = men.edition_count.order(ascending=True).values


        women_percentage = 0
        if (women.shape[0] > 0):
            women_percentage = [women[women.edition_count == x].shape[0]/float(women.shape[0]) for x in range(1, max_num_editions)]
            self.logfile.write("\n\n % local women ")
            self.logfile.write(str(women[women.edition_count < 2].shape[0]/float(women.shape[0])))
            self.logfile.write("\n percentage women %s"%str(len(women.index)/float((len(women.index)+len(men.index)))))
            women_vals = women.edition_count.order(ascending=True).values



        if (women.shape[0] > 0 and men.shape[0] > 0):
            j_m = 0
            j_w = 0

            U, p =  stats.mstats.mannwhitneyu(women.edition_count, men.edition_count)
            ut.write_mannwithneyu(U, p, women.edition_count, men.edition_count, self.logfile)

            #for i in topk:
            #    perc_men = len(men.edition_count.index) * i/100.0
            #    print "bottom %s percent of men are %s  - mean %s - median %s"% (str(i), str(perc_men), np.mean(men_vals[:int(perc_men)]), np.median(men_vals[:int(perc_men)]))
            #    top_men.append(men_vals[:int(perc_men)])
            #    perc_women = len(women.edition_count.index) * i/100.0
            #    print "bottom %s percent of women are %s  - mean %s - median %s"% (str(i), str(perc_women), np.mean(women_vals[:int(perc_men)]), np.median(women_vals[:int(perc_men)]))
            #    top_women.append(women_vals[:int(perc_women)])
            #    j_m += int(perc_men)
            #    j_w +=  int(perc_women)

            #ut.plotTopk(top_women, top_men, ['pink', 'blue'], topk, path+'/numedition_gender_topk.png')
            ut.plot_percentage(women_percentage, men_percentage, ['pink', 'blue'], range(1, max_num_editions), path+'/numedition_gender_percentage'+self.pre+"-"+self.post+'.png')
            self.genderBoxplots(women, men, labels, path)

            self.logfile.write("\n\n num women %s"%len(women.index))
            self.logfile.write("\n num men %s"%len(men.index))

            data = [women.edition_count.values, men.edition_count.values]

            # Compute the qth percentile of women and men
            q75 = [np.percentile(x, q=75) for x in data]

            self.logfile.write("\n third quartil (75th percentile):  "+str(q75))
            q95 = [np.percentile(x, q=95) for x in data]
            self.logfile.write("\n 95th percentile: "+str(q95))
            q99 = [np.percentile(x, q=99) for x in data]
            self.logfile.write("\n 99th percentile: "+str(q99))

            q99_women = q99[0]
            q99_men = q99[1]
            self.logfile.write("\n threshold women 99th percentile: %s"%q99_women)
            self.logfile.write("\n threshold men 99th percentile: %s"%q99_men)
            men_percentage = 0
            th = np.min(q99)


            if (men.shape[0] > 0):
                men_percentage = [men[men.edition_count == x].shape[0]/float(men.shape[0]) for x in range(1, int(th))]
            women_percentage = 0
            if (women.shape[0] > 0):
                women_percentage = [women[women.edition_count == x].shape[0]/float(women.shape[0]) for x in range(1, int(th))]

            ut.plot_percentage(women_percentage, men_percentage, ['pink', 'blue'], range(1, int(th)), path+'/numedition_gender_percentage'+self.pre+"-"+self.post+'_99.png')


            # RANDOM BASELINE FOR RATIO
            fake_ratios_norm = list()
            fake_ratios = list()
            for i in range(1, 1000):
                #print (people.gender.value_counts())
                people["random_gender"] = pd.Series(np.random.permutation(people.gender.values), index=people.index)
                #print (people.random_gender.value_counts())

                fake_men = people[people.random_gender =="male"]
                fake_women = people[people.random_gender =="female"]
                item_frequency_fake_female = itemfreq(np.array(fake_women['edition_count'].values.tolist()))
                item_frequency_fake_male = itemfreq(np.array(fake_men['edition_count'].values.tolist()))
                fake_ratios_norm.append(self.get_ratio(item_frequency_fake_female, item_frequency_fake_male, True))
                fake_ratios.append(self.get_ratio(item_frequency_fake_female, item_frequency_fake_male, True))


            item_frequency_female = itemfreq(np.array(women['edition_count'].values.tolist()))
            item_frequency_male = itemfreq(np.array(men['edition_count'].values.tolist()))
            ratio_norm = self.get_ratio(item_frequency_female, item_frequency_male, True)

            mean_fake_ratio_norm = {}
            mean_fake_ratio = {}
            for key in ratio_norm.keys():
                vals = []
                vals_norm = []
                for dic1 in fake_ratios_norm:
                    if key in dic1:
                        vals_norm.append(dic1.get(key))
                mean_fake_ratio_norm[key] = np.mean(vals_norm)
                for dic2 in fake_ratios:
                    if key in dic2:
                        vals.append(dic2.get(key))
                mean_fake_ratio[key] = np.mean(vals)

            # if we plor normalized ratio we should take the log since otherwise upper boun dis 0 but ratio can become extremly small.
            #ut.plotratio(ratio_norm, mean_fake_ratio_norm, ['g','r--'], ['empirical gender', 'random gender'], self.pre+"-"+self.post, path+'/numedition_gender_ratio'+self.pre+"-"+self.post+'_norm.png', 'Num Editions', 'Male Proportion/Female Proportion', False, False)
            ratio = self.get_ratio(item_frequency_female, item_frequency_male, False)
            lowess = sm.nonparametric.lowess(ratio.values(), ratio.keys(), frac=0.1)
            ut.plotratio(ratio, lowess, mean_fake_ratio,  ['b^','g', 'r--'], ['empirical gender', 'lowess fit', 'random gender'], self.pre+"-"+self.post, path+'/numedition_gender_ratio'+self.pre+"-"+self.post+'.png', 'Num Editions', 'Male/Female', False, False)


            #ratio = self.get_ratio(item_frequency_female, item_frequency_male, True)
            #ut.plotline(list(ratio.keys()), list(ratio.values()),  ['pink','blue'], path+'/numedition_gender_ratio'+self.pre+'_norm.png', 'Num Editions', 'Female-Male-Ratio', False, False)
            #ratio = self.get_ratio(item_frequency_female, item_frequency_male, False)
            #ut.plotline(list(ratio.keys()), list(ratio.values()),  ['pink','blue'], path+'/numedition_gender_ratio'+self.pre+'.png', 'Num Editions', 'Female-Male-Ratio', False, False)


            #ut.plot_rank_size(list([item_frequency_female[:np.max(q99)], item_frequency_male[:np.max(q99)]]), labels, ['pink','blue'], path+'/numedition_gender_ranksize_99.png', 'Rank', 'Num Editions', False, True)

            #print "Mann Withney U Test Frequ Dist:"
            #print stats.mstats.mannwhitneyu(item_frequency_female, item_frequency_male)
            #print stats.ranksums(item_frequency_female, item_frequency_male)


            ut.plot_cdf(list([item_frequency_female, item_frequency_male]), labels, ['pink','blue'], path+'/numedition_gender_ccdf'+self.pre+"-"+self.post+'.png', 'Num Editions', True, False, True)
            #ut.plot_cdf(list([item_frequency_female[:np.max(q95)], item_frequency_male[:np.max(q95)]]), labels, ['pink','blue'], path+'/numedition_gender_ccdf_95.png', 'Num Editions', True, False, True)
            #ut.plot_cdf(list([item_frequency_female[:np.max(q99)], item_frequency_male[:np.max(q99)]]), labels, ['pink','blue'], path+'/numedition_gender_ccdf_99.png', 'Num Editions', True, False, True)


            self.logfile.write("\n\n men median(men mean), women median (women mean)")
            self.logfile.write("\n "+ str(np.median(men.edition_count.values))+'('+str(np.mean(men.edition_count.values))+'), '+ str(np.median(women.edition_count.values))+'('+str(np.mean(women.edition_count.values))+')')

            return {"class":classname, "median-men": np.median(men.edition_count.values),  "mean-men":np.mean(men.edition_count.values), "sem-men":stats.sem(men.edition_count.values),
            "sem-women":stats.sem(women.edition_count.values), "median-women":np.median(women.edition_count.values), "mean-women":np.mean(women.edition_count.values)}



    def regression(self):
        from statsmodels.formula.api import glm
        from statsmodels.api import families

        self.people.rename(columns={'class': 'dbpedia_class'}, inplace=True) # all_bios is the dataframe with the consolidated data. somehow it doesn't work if the class column is named "class"


        people = self.people[(self.people.birth_century >= 0) & (self.people.birth_century <= 2000)]


        m = glm("edition_count ~ C(gender,Treatment(reference='male')) + C(available_english) + C(dbpedia_class,Treatment(reference='http://dbpedia.org/ontology/Person')) + C(birth_century)",
                data=people, family=families.NegativeBinomial()).fit()

        print (m.summary(), file=self.logfile) # <-- this gives you the table of coefficients with p-values, confidence intervals, and so on



if __name__ == "__main__":

    # analyze all data without restricting start and end by birth year
    pop = GlobalWikipediaPopularity('data/consolidated_person_data.csv', -1, -1)
   # pop.analyze_numlangedition_per_profession()
   # pop.analyze_numlangedition_per_decade()

    # select interval in which people should be born
    startyears  = [1900]
    endyear = 2015
    for year in startyears:
        pop = GlobalWikipediaPopularity('data/consolidated_person_data.csv', year, endyear)
        print (pop.people.shape)
        pop.analyze_num_lang_edition("all", pop.people)
        pop.regression()






