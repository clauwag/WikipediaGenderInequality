
__author__ = 'wagnerca'
import pylab as plt
import numpy as np
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

import seaborn as sns
import pandas as pd
sns.set_context("poster", font_scale=1.1)


def plot_topk(top_women,top_men,colors, xaxis_values, filename):
    """
    plots the percentage of people who show up in N language editions
    :param men:
    :param women:
    :param colors:
    :return:
    """
    f, axarr  = plt.subplots(5, 2, sharex=True) #sharey='row'
    plt.xlabel("Gender")
    plt.ylabel("Num Languages")

    row = 0
    col = 0
    print xaxis_values
    for i in range(0, len(xaxis_values)):
        alldata = np.concatenate((top_women[i],top_men[i]), axis=0)
        #print alldata

        bp = axarr[row, col].boxplot([top_women[i], top_men[i]], sym="") #notch=0, sym='+', vert=1, whis=1.5
        #sns.boxplot(data=[top_women[i], top_men[i]], sym = "", ax = axarr[row, col])
        #sns.stripplot(data=alldata, size=4, jitter=True, edgecolor="gray", ax = axarr[row, col])

        #plt.setp(bp['boxes'], color='black')
        #plt.setp(bp['whiskers'], color='black')
        #plt.setp(bp['fliers'], color='red', marker='+')
        axarr[row, col].annotate("bottom "+str(xaxis_values[i])+"%", xy=(0.9, 0.7), xycoords='axes fraction', fontsize=10,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom')
        #axarr[row, col].set_ylim(np.min(alldata), np.max(alldata))
        axarr[row, col].set_xticks([1, 2], ['women', 'men'])

        print "\n bottom "+str(xaxis_values[i])+"%"
        print (row, col)
        print "min %s, max %s"%(np.min(alldata), np.max(alldata))
        print "mean women %s (median women %s), mean men %s (median men %s)" % (str(np.mean(top_women[i])), str(np.median(top_women[i])), str(np.mean(top_men[i])), str(np.median(top_men[i])))


        row += 1
        if(row >= 5):
            row = 0
            col += 1
    f.savefig(filename, bbox_inches='tight')
    plt.close()




# This function takes an array of numbers and smoothes them out.
# Smoothing is useful for making plots a little easier to read.

def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


def plot_smooth_lines(years, mean_vals, sem_vals, filename):
    mean_PlyCount = sliding_mean(mean_vals,
                                 window=10)
    sem_PlyCount = sliding_mean(sem_vals.mul(1.96).values,
                            window=10)

    # You typically want your plot to be ~1.33x wider than tall.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    plt.ylim(63, 85)

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(range(1850, 2011, 20), fontsize=14)
    plt.yticks(range(65, 86, 5), fontsize=14)

    # Along the same vein, make sure your axis labels are large
    # enough to be easily read as well. Make them slightly larger
    # than your axis tick labels so they stand out.
    plt.ylabel("Ply per Game", fontsize=16)

    # Use matplotlib's fill_between() call to create error bars.
    # Use the dark blue "#3F5D7D" as a nice fill color.
    plt.fill_between(years, mean_PlyCount - sem_PlyCount,
                     mean_PlyCount + sem_PlyCount, color="#3F5D7D")

    # Plot the means as a white line in between the error bars.
    # White stands out best against the dark blue.
    plt.plot(years, mean_PlyCount, color="white", lw=2)

    plt.savefig(filename, bbox_inches="tight");


def plot_shaded_lines(my_xticks, y1, y2, error1, error2, ylab, xlab, filename):
    plt.figure(figsize=(6,6))
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    x = range(0, len(y1))
    plt.plot(x, y1, 'k-', color="blue",  label='men')
    plt.fill_between(x, y1-error1, y1+error1, facecolor='blue', alpha=.2)

    plt.plot(x, y2, 'k-', color="red",  label='women')
    plt.fill_between(x, y2-error2, y2+error2, facecolor='red', alpha=.2)

    #if isinstance(x, (int, long, float, complex)):
    #    plt.xlim(np.min(x), np.max(x))
    plt.gcf().subplots_adjust(bottom=0.3)

    plt.xticks(x, my_xticks)
    plt.xticks(rotation=70, fontsize=14)
    plt.yticks(fontsize=14)
    #plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=14)
    plt.ylabel(ylab, fontsize=14)
    plt.xlabel(xlab, fontsize=14)
    plt.legend()

    plt.savefig(filename)



def plot_percentage(women_percentage, men_percentage, colors, xaxis_values, filename):
    """
    plots the percentage of people who show up in N language editions
    :param men:
    :param women:
    :param colors:
    :return:
    """
    fig = plt.figure(figsize=(6,6))
    if len(colors) == 0:
        plt.gca().set_color_cycle(['pink', 'blue', 'yellow', 'red', 'black'])
    else:
        plt.gca().set_color_cycle(colors)
    #print xaxis_values
    #print women_percentage
    #print len(xaxis_values)
    #print len(women_percentage)
    plt.plot(xaxis_values, women_percentage[:len(xaxis_values)], linewidth=1)
    plt.plot(xaxis_values, men_percentage[:len(xaxis_values)], linewidth=1)
    plt.xlabel("Log Num Editions")
    plt.ylabel("Log Proportion")
    plt.xscale("log")
    plt.yscale("log")
    #plt.ysca:len(xaxis_values)lotle("log")
    plt.savefig(filename)
    plt.close()


def plot_cdf(_itemFreqList, labels, colors, path, xlabel, plotCCDF, xlog, ylog):
    """
    :param _itemFreqList: list of 2D array which consist of items and frequency of that item
            labels: list of names for the lists
            colors: colors of the lists
            path: where to save the plot
            xlabel: name of xaxis; y-axis is always ccdf
            xlog: boolean indicating if x axis is log or not
    :return:
    """
    fig = plt.figure(figsize=(6,6))
    if len(colors) == 0:
        plt.gca().set_color_cycle(['pink', 'blue', 'yellow', 'red', 'black'])
    else:
        plt.gca().set_color_cycle(colors)


    for _itemFreq in _itemFreqList:
        _itemFreq_sorted = np.array(sorted(_itemFreq,key=lambda x: x[0]))

        numberOfFrequency_sorted = _itemFreq_sorted[:, 1]
        numberOfReviews_sorted = _itemFreq_sorted[:,0]

        cdf = numberOfFrequency_sorted.cumsum()/ float(numberOfFrequency_sorted.sum())

        if plotCCDF==1:
            ccdf= 1-cdf
            sign = "\geq"
        else:
            ccdf = cdf
            sign = "<"

        plt.plot(numberOfReviews_sorted, ccdf, linewidth=1)

    #plt.title("ccdf plot")
    if xlog==1:
        plt.xscale('log')
        plt.xlabel('$log \/ '+xlabel+' \/ x $')
    else:
        plt.xlabel('$'+xlabel+' \/ x $')
    if ylog == 1:
        plt.yscale('log')
        plt.ylabel("$ log \/ P(X "+sign+" x) $")
    else:
        plt.ylabel("$ P(X "+sign+" x)$")

    plt.legend(labels, loc='upper right')
    plt.grid(False)
    plt.savefig(path)


#def plot_rank_size(_itemFreqList, labels, colors, path, xlabel, ylabel, xlog, ylog):
    #TODO


def plot_facet_dist(data, facet_col, data_col, filename):
    fig = plt.figure(figsize=(6,6))
    g = sns.FacetGrid(data, col=facet_col, size=4, aspect=1.5)
    g.map(sns.distplot, data_col, kde=False)
    g.set(yscale='log')
    g.add_legend()
    plt.grid(False)
    plt.savefig(filename)
    plt.close()



def plot_numedition_size(_itemFreqList, labels, colors, path, xlabel, ylabel, xlog, ylog):

    fig = plt.figure(figsize=(6,6))


    if len(colors) == 0:
        plt.gca().set_color_cycle(['pink', 'blue', 'yellow', 'red', 'black'])
    else:
        plt.gca().set_color_cycle(colors)

    numberOfFrequency_sorted = {}
    for _itemFreq in _itemFreqList:
        _itemFreq_sorted = np.array(sorted(_itemFreq,key=lambda x: x[0]))

        numberOfFrequency_sorted = _itemFreq_sorted[:, 1]
        numberOfReviews_sorted = _itemFreq_sorted[:,0]


        plt.plot(numberOfReviews_sorted, numberOfFrequency_sorted, linewidth=1)

    #plt.title("ccdf plot")
    if xlog==1:
        plt.xscale('log')
        plt.xlabel('$log \/ '+xlabel+'  $')
    else:
        plt.xlabel(xlabel)
    if ylog == 1:
        plt.yscale('log')
        plt.ylabel("$ log "+ylabel)
    else:
        plt.ylabel(ylabel)

    plt.legend(labels, loc='upper right')
    plt.grid(False)
    plt.savefig(path, bbox_inches='tight')
    plt.close()



def plotratio(ratio, ratiofit, fakeratio, styles, labels, title, path, xlabel, ylabel, xlog, ylog):
    xval = list(ratio.keys())
    yval = list(ratio.values())
    fit_xval = list(ratiofit[:, 0])
    fit_yval = list(ratiofit[:, 1])

    fxval = list(fakeratio.keys())
    fyval = list(fakeratio.values())

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(False)

    plt.title(title, loc='left', color='r')
    #plt.text(0.5, 0.5, title,
    # horizontalalignment='center',
    # verticalalignment='center',
    # transform = ax.transAxes)
    plotline(xval, yval, styles[0], path, xlabel, ylabel, xlog, ylog)
    plotline(fit_xval, fit_yval, styles[1], path, xlabel, ylabel, xlog, ylog)
    plotline(fxval, fyval, styles[2], path, xlabel, ylabel, xlog, ylog)
    ax.set_ylim([-1, np.max(yval)+3.5])
    ax.set_xlim([0, np.max(xval)])
    ax.legend(numpoints=3, scatterpoints=3)
    ax.legend(labels, loc='upper right')
    #plt.grid(False)
    plt.savefig(path, bbox_inches='tight')


def plotline(xvals, yvals, style, path, xlabel, ylabel, xlog, ylog):

    plt.plot(xvals, yvals, style, linewidth=2, markersize=5)

    #plt.title("ccdf plot")
    if xlog==1:
        plt.xscale('log')
        plt.xlabel('log \/ '+xlabel+'  ')
    else:
        plt.xlabel(xlabel)
    if ylog == 1:
        plt.yscale('log')
        plt.ylabel("log "+ylabel)
    else:
        plt.ylabel(ylabel)






def rank_size_plot(data, attr, label, filename):
    plt.figure(figsize=(6,6))

    plt.subplot(111)

    men = data[data.gender == 'male']
    rank_m = men[attr].sort(ascending=False, inplace=False)
    women = data[data.gender == 'female']
    rank_w = women.edition_count.sort(ascending=False, inplace=False)

    plt.plot(np.arange(rank_m.shape[0], dtype=np.float) / rank_m.shape[0] + 0.001, rank_m, label='Men', linestyle='none', marker='.', alpha=0.5)
    plt.plot(np.arange(rank_w.shape[0], dtype=np.float) / rank_w.shape[0] + 0.001, rank_w, label='Women', linestyle='none', marker='.', alpha=0.5)
    plt.xlabel('Normalized Rank')
    plt.ylabel(label)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    plt.xlim([0.001, 1.001])
    plt.savefig(filename, bbox_inches='tight')




def write_mannwithneyu(U, p, arr1, arr2, logfile):
    gsize1 = len(arr1)
    gsize2 = len(arr2)
    prob = U/float(gsize1*gsize2)
    if(p < 0.05):
        logfile.write("\n U=%s, p=%s, S1=%s, S2=%s, median1(mean1)=%s(%s), median2(mean2)=%s(%s), U/S1*S2 = %s  ******" % (U, p, gsize1, gsize2, np.median(arr1), np.mean(arr1), np.median(arr2), np.mean(arr2), prob))
        logfile.write("\n U statistic divide by the product of the two sample sizes --> probability that a score randomly drawn from population A will be greater than a score randomly drawn from population B is %s"% str(U/float(gsize1*gsize2)))
    else:
        logfile.write("\n  U=%s, p=%s, S1=%s, S2=%s, median1(mean1)=%s(%s), median2(mean2)=%s(%s), U/S1*S2 = %s " %  (U, p, gsize1, gsize2, np.median(arr1), np.mean(arr1), np.median(arr2), np.mean(arr2), prob))
        logfile.write("\n probability that a score randomly drawn from population A will be greater than a score randomly drawn from population B is %s"% str(U/float(gsize1*gsize2)))




#if __name__ == '__main__':
#    #test cdf function
#    from scipy.stats import itemfreq
#    item_frequency_female = itemfreq(np.array([2, 2, 2, 2, 2, 3, 3, 5, 6, 8]))
#    item_frequency_male = itemfreq(np.array([1, 1,1 , 1, 1, 1, 1, 1, 4, 4, 5]))
#    print item_frequency_female
#    print item_frequency_male
#    plot_cdf([item_frequency_female, item_frequency_male], ['female', 'male'], ['pink', 'blue'], "./test_ccdf.png", "num", 1, 0, 0)
