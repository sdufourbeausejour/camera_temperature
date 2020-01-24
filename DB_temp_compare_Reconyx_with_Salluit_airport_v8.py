# -*- coding: Latin-1 -*-
"""

@author: Administrateur
"""

### Modules ##################################################################
import os                         # pour pouvoir récupérer le nom du script
import numpy as np
import scipy as sp
import datetime
from datetime import date       # manipulation de dates
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy import stats
import sklearn
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, DateFormatter

mpl.rcdefaults()
mpl.rcParams["mathtext.default"]= "regular"
mpl.rcParams["font.size"] = 12. # change the size of the font in every figure
mpl.rcParams["font.family"] = "Arial" # font Arial in every figure
mpl.rcParams["font.weight"] = 100 # font Arial in every figure
mpl.rcParams["axes.labelsize"] = 12.
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.linewidth"] = 0.6 # thickness of the axes lines
mpl.rcParams["pdf.fonttype"] = 42  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
                                   # editing the text in illustrator

script_name = os.path.basename(__file__)

def get_date(year,month,day,hour,minute,second):
    date = mpl.dates.date2num(datetime.datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(second)))
    return date

v_get_date = np.vectorize(get_date)

def get_date_from_mpl(mpl_date):
    date = mpl.dates.num2date(mpl_date)
    return date
v_get_date_from_mpl = np.vectorize(get_date_from_mpl)
v_date2num = np.vectorize(lambda x: mpl.dates.date2num(pd.Timestamp(x).to_pydatetime()))

# Load data
data = {}
data["DB"] = pd.read_csv("data/DeceptionBay_Reconyx_assembled.txt", sep=";", comment='#')
names = ["year","month","day","hour","minute","second","temperature"]
for key in data.keys():
    data[key].columns = names
data["EC"] = pd.read_csv("data/EC/S_EC_temperatures_assembled.txt", header=13)
data["EC"]["year"] = data["EC"]["Année"]
data["EC"]["month"] = data["EC"]["Mois"]
data["EC"]["day"] = data["EC"]["Jour"]
data["EC"]["hour"] = data["EC"]["Heure"]
data["EC"]["minute"] = [x[3:] for x in data["EC"]["hour"]]
data["EC"]["hour"] = [x[0:2] for x in data["EC"]["hour"]]
data["EC"]["second"] = np.zeros(len(data["EC"]["hour"]))
data["EC"]["temperature"] = [str.replace(str(x), ",", ".") for x in data["EC"]["Temp (°C)"]]

dates = {}
T = {}
for i, camera in enumerate(data.keys()):
    # Get dates array
    d = data[camera]
    df = d[['year','month','day','hour','minute','second']].copy()
    year = df['year'].values
    month = df['month'].values
    day = df['day'].values
    hour = df['hour'].values
    minute = df['minute'].values
    second = df['second'].values
    dates[camera] = v_get_date(year,month,day,hour,minute,second)
    T[camera] = d['temperature']

# Compute diff between BD and Salluit T
# daily mean
daily_mean = {}
for camera in ["DB","EC"]:
    daily_mean[camera] = pd.DataFrame(np.transpose(np.vstack([v_get_date_from_mpl(dates[camera]),T[camera]])))
    daily_mean[camera].columns=["dates","T"]
    daily_mean[camera] = daily_mean[camera].set_index("dates")
    daily_mean[camera] = daily_mean[camera]["T"]
    daily_mean[camera] = daily_mean[camera].dropna()
    daily_mean[camera] = daily_mean[camera].sort_index()
    daily_mean[camera] = pd.to_numeric(daily_mean[camera],errors='coerce')
    daily_mean[camera] = daily_mean[camera].resample("D").mean()
    # print("NaN in "+camera+":")
    # print(len(np.where(np.isnan(daily_mean[camera]))[0]))
    # daily_mean[camera] = daily_mean[camera].interpolate(method="linear")
    # print(daily_mean[camera])

intersection = pd.concat([daily_mean["DB"].dropna(), daily_mean["EC"].dropna()], axis=1, join='inner')
intersection.columns = ["DB","EC"]
intersection["diff"] = intersection["DB"] - intersection["EC"]
intersection["corrected"] = intersection["DB"]
figures_path = "../../figures/special_issue_TSX/R1/"+script_name[0:-3]+".pdf"
fig, axes = plt.subplots(3, 2, figsize=(10, 6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.1, top=0.92, hspace=0.1, wspace=0.2) # adjust the box of axes regarding the figure size
axes = axes.flatten()
# fig.suptitle("Comparing Deception Bay Reconyx Temperature Measurements with Salluit Airport", fontsize=12)
for i, year in enumerate([2015,2016,2017]):
    xmin = mpl.dates.date2num(datetime.datetime(year=int(year),month=int(9),day=int(15)))
    xmax = mpl.dates.date2num(datetime.datetime(year=int(year),month=int(11),day=int(30)))

    # Plot each series
    axes[i*2].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis
    axes[i*2].yaxis.set_ticks(np.arange(-30,30,10))
    axes[i*2].yaxis.set_ticks(np.arange(-40,40,2), minor=True)
    axes[i*2].set_ylim(-40,30)
    axes[i*2].set_ylabel(r"T ($^\circ$C)")
    axes[i*2].axhline(y=0, ls="-", c="k", linewidth=0.5)
    axes[i*2].annotate(str(year)+"-"+str(year+1), xy=(0.03,0.85), xycoords="axes fraction",fontsize=12,color="k")
    axes[i*2].plot_date(daily_mean["DB"].index, daily_mean["DB"], linewidth=0.4,marker=" ",linestyle="-",color="k")
    axes[i*2].plot_date(daily_mean["EC"].index, daily_mean["EC"], linewidth=0.4,marker=" ",linestyle="-",color="r")
    if i == 1:
        axes[i*2].annotate(r"Camera", xy=(0.03,0.18), xycoords="axes fraction",fontsize=12,color="k")
        axes[i*2].annotate(r"Airport (50 km)", xy=(0.03,0.08), xycoords="axes fraction",fontsize=12,color="r")

    # Compute correlation coefficient
    joined = pd.concat([daily_mean["DB"], daily_mean["EC"]], axis=1, join='inner')
    joined.columns = ["Reconyx", "EC"]
    joined = joined.dropna()
    joined = joined.loc[mpl.dates.num2date(xmin):mpl.dates.num2date(xmax)]
    pearson = scipy.stats.pearsonr(joined["Reconyx"], joined["EC"])
    print(pearson)
    axes[i*2].annotate(r"r = {:.2f}".format(pearson[0]), xy=(0.7, 0.18), xycoords="axes fraction", fontsize=12,
                     color="k")

    difference = joined["Reconyx"]-joined["EC"]
    MSE = np.average([x**2 for x in difference])
    print(np.sqrt(MSE))
    axes[i*2].annotate(r"RMSE = {:.1f}$\degree$C".format(np.sqrt(MSE)), xy=(0.7, 0.08), xycoords="axes fraction", fontsize=12,
                     color="k")

    # Plot difference
    year_data_ind = (intersection["diff"].index > mpl.dates.num2date(xmin)) & (intersection["diff"].index <= mpl.dates.num2date(xmax))
    yearly_diff = intersection["diff"][year_data_ind]
    axes[i*2 +1].plot_date(v_date2num(yearly_diff.index),yearly_diff, linewidth=0.4 ,marker=" ",linestyle="-",color="k")
    axes[i*2 +1].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis
    axes[i*2 +1].axhline(y=0, ls="-", c="k", linewidth=0.5)

    # # Model bias
    # mu = mpl.dates.date2num(datetime.datetime(year+1,5,1))
    # sigma = 50
    # gaussian = scipy.stats.norm.pdf(v_date2num(yearly_diff.index), mu, sigma)
    # bias = 4.0*gaussian/max(gaussian)
    # axes[i+3].plot_date(v_date2num(yearly_diff.index),bias, linewidth=1 ,marker=" ",linestyle="-",color="b")
    #
    # pearson = scipy.stats.pearsonr(yearly_diff, bias)
    # print(pearson)
    # axes[i+3].annotate(r"r = {:.2f}".format(pearson[0]), xy=(0.68, 0.18), xycoords="axes fraction", fontsize=12,
    #                  color="k")
    #
    # difference = yearly_diff - bias
    # MSE = np.average([x**2 for x in difference])
    # axes[i+3].annotate(r"RMSE = {:.1f}$\degree$C".format(np.sqrt(MSE)), xy=(0.68, 0.08), xycoords="axes fraction", fontsize=12,)

    # if i == 0:
    #     axes[i+3].annotate(r"Modeled camera bias", xy=(0.05,0.08), xycoords="axes fraction",fontsize=12,color="b")
    axes[i*2 +1].set_ylabel(r"$\Delta T$ ($^\circ$C)")
    axes[i*2 +1].yaxis.set_ticks(np.arange(-5,15,5))
    axes[i*2 +1].yaxis.set_ticks(np.arange(-10,15,1), minor=True)
    axes[i*2 +1].set_ylim(-6,12)

# reccurent setup
for ax in axes:
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(7)
    for tick in ax.yaxis.get_major_ticks():
        tick.set_pad(8)
    for tick in ax.yaxis.get_major_ticks():
        tick.set_pad(8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    # xtick formatting
    days = DayLocator()  # every day
    months = MonthLocator()  # every month
    years = YearLocator()   # every year
    monthsFmt = DateFormatter("%b") # formatting date
    yearsFmt = DateFormatter("%Y") # formatting date
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.autoscale_view()

    tick_list = list()
    for year in np.arange(2015,2019):
        for month in np.arange(1,13):
            for week_day in [8,15,22,29]:
                if not (month == 2 and week_day == 29):
                    tick_list.append(mpl.dates.date2num(datetime.datetime(year=year,month=month,day=week_day)))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_list))
    ax.tick_params(direction='in',which="both",right=1,top=0)

# for i in list([1,2,4,5]):
#     axes[i].get_yaxis().set_ticklabels([])
#     axes[i].set_ylabel("")

for i in list([0,1,2,3]):
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].set_xlabel("")

#plt.show()
fig.savefig(figures_path)
fig.savefig(figures_path[0:-4]+".png",transparent=False, dpi=300)
plt.close()
