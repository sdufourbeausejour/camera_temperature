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
from scipy.interpolate import interp1d
from scipy import stats
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def month_name_to_month(x):
    """from "Jan" to 1"""
    month_number = np.zeros(len(x))
    month_number[np.where(x == "Jan")] = 1
    month_number[np.where(x == "Feb")] = 2
    month_number[np.where(x == "Mar")] = 3
    month_number[np.where(x == "Apr")] = 4
    month_number[np.where(x == "May")] = 5
    month_number[np.where(x == "Jun")] = 6
    month_number[np.where(x == "Jul")] = 7
    month_number[np.where(x == "Aug")] = 8
    month_number[np.where(x == "Sep")] = 9
    month_number[np.where(x == "Oct")] = 10
    month_number[np.where(x == "Nov")] = 11
    month_number[np.where(x == "Dec")] = 12
    return month_number

def cloudiness_factor(x):
    """get cloudiness_factor for a time series in the Arctic, from Parkinson and Washington 1979"""
    # cloudiness factor, 0.50 for December through March,
    # 0.55 for April, 0.70 for May, 0.75 for June and July,
    # 0.80 for August and September, 0.70 for October, 0.60 for November
    x = x.astype('datetime64[M]').astype(int) % 12 + 1
    cloudiness_factor = np.zeros(len(x))
    cloudiness_factor[np.where((x >= 1) & (x <= 3))] = 0.50
    cloudiness_factor[np.where(x == 4)] = 0.55
    cloudiness_factor[np.where(x == 5)] = 0.70
    cloudiness_factor[np.where((x >= 6) & (x <= 7))] = 0.75
    cloudiness_factor[np.where((x >= 8) & (x <= 9))] = 0.80
    cloudiness_factor[np.where(x == 10)] = 0.70
    cloudiness_factor[np.where(x == 11)] = 0.60
    cloudiness_factor[np.where(x == 12)] = 0.50
    return cloudiness_factor

# Load data
data = {}
data["Nord_200"] = pd.read_csv("data/Quaqtaq_Reconyx_Nord_200.txt", sep=";", comment='#')
data["Nord_268"] = pd.read_csv("data/Quaqtaq_Reconyx_Nord_268.txt", sep=";", comment='#')
data["Hearn_200"] = pd.read_csv("data/Quaqtaq_Reconyx_Hearn_200.txt", sep=";", comment='#')
names = ["year","month","day","hour","minute","second","temperature"]
for key in data.keys():
    data[key].columns = names
# Load EC data
data["EC"] = pd.read_csv("data/EC/Q_EC_temperatures_assembled.txt", comment='#')
data["EC"]["minute"] = [x[3:] for x in data["EC"]["hour"]]
data["EC"]["hour"] = [x[0:2] for x in data["EC"]["hour"]]
data["EC"]["second"] = np.zeros(len(data["EC"]["hour"]))
data["EC"]["temperature"] = [str.replace(str(x), ",", ".") for x in data["EC"]["temperature"]]
# read sunrise data to plot
data["sunrise"] = pd.read_csv("data/NRCC/2015_2018_sunrise_sunset.txt", sep=";", comment='#')
data["sunrise"]["hour"] = [x.split(":")[0] for x in data["sunrise"]["sunrise"]]
data["sunrise"]["minute"] = [x.split(":")[1] for x in data["sunrise"]["sunrise"]]
data["sunrise"]["second"] = np.zeros(len(data["sunrise"]["hour"]))
data["sunrise"]["month"] = month_name_to_month(data["sunrise"]["month"])
data["sunset"] = pd.read_csv("data/NRCC/2015_2018_sunrise_sunset.txt", sep=";", comment='#')
data["sunset"]["hour"] = [x.split(":")[0] for x in data["sunset"]["sunset"]]
data["sunset"]["minute"] = [x.split(":")[1] for x in data["sunset"]["sunset"]]
data["sunset"]["second"] = np.zeros(len(data["sunset"]["hour"]))
data["sunset"]["month"] = month_name_to_month(data["sunset"]["month"])
# Get dates array
dates = {}
for i, camera in enumerate(data.keys()):
    d = data[camera]
    df = d[['year','month','day','hour','minute','second']].copy()
    year = df['year'].values
    month = df['month'].values
    day = df['day'].values
    hour = df['hour'].values
    minute = df['minute'].values
    second = df['second'].values
    dates[camera] = v_get_date(year,month,day,hour,minute,second)
# Get temperature
T = {}
for i, camera in enumerate(["Nord_200","Nord_268","Hearn_200","EC"]):
    d = data[camera]
    T[camera] = d['temperature']

# Compute diff between BD and Salluit T
# daily mean
daily_mean = {}
for camera in ["Nord_268","EC"]:
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

intersection = pd.concat([daily_mean["Nord_268"].dropna(), daily_mean["EC"].dropna()], axis=1, join='inner')
intersection.columns = ["Nord_268","EC"]
intersection["diff"] = intersection["Nord_268"] - intersection["EC"]
figures_path = "../../figures/meteo/"+script_name[0:-3]+".pdf"
fig, axes = plt.subplots(figsize=(14, 6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.1, top=0.92, hspace=0.2, wspace=0.1) # adjust the box of axes regarding the figure size
gs = gridspec.GridSpec(3, 3,
                       width_ratios=[1,1,1],
                       height_ratios=[1,1,1]
                       )
axes = list()
axes.append(plt.subplot(gs[0,0]))
axes.append(plt.subplot(gs[0,1]))
axes.append(plt.subplot(gs[0,2]))
axes.append(plt.subplot(gs[1,0]))
axes.append(plt.subplot(gs[1,1]))
axes.append(plt.subplot(gs[1,2]))
axes.append(plt.subplot(gs[2,0]))
axes.append(plt.subplot(gs[2,1]))
axes.append(plt.subplot(gs[2,2]))

fig.suptitle("Comparing Quaqtaq Reconyx Nord Temperature Measurements with Airport", fontsize=12)

smoothed_by_year = {}
for i, year in enumerate([2015,2016,2017]):
    xmin = mpl.dates.date2num(datetime.datetime(year=int(year),month=int(9),day=int(15)))
    xmax = mpl.dates.date2num(datetime.datetime(year=int(year)+1,month=int(9),day=int(15)))
    axes[i].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis

    # Plot difference
    axes[i].axhline(y=0, ls="-", c="k", linewidth=0.5)
    axes[i].annotate(str(year)+"-"+str(year+1), xy=(0.04,0.875), xycoords="axes fraction",fontsize=12,color="k")
    axes[i].set_ylabel(r"$\Delta T_{268 - EC}$ ($^\circ$C)")
    axes[i].yaxis.set_ticks(np.arange(-10,30,1), minor=True)
    axes[i].plot_date(v_date2num(intersection["diff"].index),intersection["diff"].values, linewidth=0.4 ,marker=" ",linestyle="-",color="k")
    # Mean diff for that year
    yearly_mean_diff_ind = (intersection["diff"].index > mpl.dates.num2date(xmin)) & (intersection["diff"].index <= mpl.dates.num2date(xmax))
    yearly_mean_diff = intersection["diff"][yearly_mean_diff_ind]
    smooth_diff = savgol_filter(yearly_mean_diff.values, 121, 3) # window size, polynomial order
    axes[i].plot_date(yearly_mean_diff.index,smooth_diff, linewidth=1 ,marker="",linestyle="-",color="b")

    # save with relative date 2015-2016
    new_index = [datetime.datetime(
                    year=x.year-i,
                    month=x.month,
                    day=x.day) for x in yearly_mean_diff.index]
    # smoothed_by_year[str(year)] = pd.Series(smooth_diff,new_index)
    # all_smoothed = pd.concat([smoothed_by_year["2015"],smoothed_by_year["2016"],smoothed_by_year["2017"]],axis=1,join="outer")
    # mean_smoothed = all_smoothed.mean(axis=1)
    # mean_smoothed_again = savgol_filter(mean_smoothed.values, 121, 3) # window size, polynomial order
    new_index = np.asarray(v_date2num(new_index))
    axes[i+3].plot_date(new_index,smooth_diff, linewidth=1 ,marker="",linestyle="-",color="k")
    mu = new_index[np.where(smooth_diff == max(smooth_diff))]
    sigma = 50
    # print(v_date2num(new_index))
    gaussian = stats.norm.pdf(new_index, mu, sigma)
    axes[i+3].plot_date(new_index,4*gaussian/max(gaussian), linewidth=1 ,marker="",linestyle="-",color="r")

    year_data_ind = (intersection["diff"].index > mpl.dates.num2date(xmin)) & (intersection["diff"].index <= mpl.dates.num2date(xmax))
    year_data = intersection["diff"][year_data_ind]
    axes[i+6].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis
    axes[i+6].plot_date(v_date2num(year_data.index),year_data.values, linewidth=0.4 ,marker=" ",linestyle="-",color="k")
    axes[i+6].plot_date(v_date2num(year_data.index),year_data.values - 4*gaussian/max(gaussian), linewidth=0.4 ,marker=" ",linestyle="-",color="r")
    axes[i+6].axhline(y=0, ls="-", c="k", linewidth=0.5)

    # axes[i+3].plot_date(v_date2num(yearly_mean_diff.index),4*gaussian/max(gaussian), linewidth=1 ,marker="",linestyle="-",color="r")


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
    ax.tick_params(direction='in',which="both",right=1,top=1)

for i in list([1,2,4,5,7,8]):
    axes[i].get_yaxis().set_ticklabels([])
    axes[i].set_ylabel("")

for i in list([0,1,2,3,4,5]):
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].set_xlabel("")

#plt.show()
fig.savefig(figures_path[0:-3]+"png", dpi=300)
plt.close()
