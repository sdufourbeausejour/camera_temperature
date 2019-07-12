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
import pandas as pd
from pandas.plotting import autocorrelation_plot
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
fig, axes = plt.subplots(2, 1, figsize=(14, 6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.1, top=0.92, hspace=0.2, wspace=0.1) # adjust the box of axes regarding the figure size
axes = axes.flatten()
fig.suptitle("Comparing Quaqtaq Reconyx Nord Temperature Measurements with Airport", fontsize=12)

# autocorrelation_plot(intersection["diff"])

# def model(x, y0):
# """Model for camera-heating induced error - diff between camera and reference"""
#     model =

#plot day duration in hours
td = v_get_date_from_mpl(dates["sunset"]) - v_get_date_from_mpl(dates["sunrise"])
day_duration_x = np.arange('2015-01', '2019-01', dtype='datetime64[D]')
day_duration = [x.seconds/3600.0 for x in td]
day = [y.astype('datetime64[D]') - y.astype('datetime64[M]') + 1 for y in day_duration_x]
day = np.asarray([int(y/np.timedelta64(1, 'D')) for y in day])
x_at_15 = day_duration_x[np.where(day == 15)]
cloudiness_at_15 = cloudiness_factor(x_at_15)

# interp every day
cloudiness = np.interp(day_duration_x.astype('datetime64[D]').astype(int),
            x_at_15.astype('datetime64[D]').astype(int), cloudiness_at_15)
smooth_cloudiness = savgol_filter(cloudiness, 121, 3) # window size, polynomial order

xmin = mpl.dates.date2num(datetime.datetime(year=2015,month=int(9),day=int(15)))
xmax = mpl.dates.date2num(datetime.datetime(year=2018,month=int(9),day=int(15)))
axes[0].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis
# axes[0].yaxis.set_ticks(np.arange(-30,30,10))
# axes[0].yaxis.set_ticks(np.arange(-40,40,2), minor=True)
# axes[0].set_ylabel(r"Temperature ($^\circ$C)")
# axes[0].axhline(y=0, ls="-", c="k", linewidth=0.5)
axes[0].annotate(str(year)+"-"+str(year+1), xy=(0.04,0.875), xycoords="axes fraction",fontsize=12,color="k")

# Plot day_duration
axes[0].set_ylim(0,1)
day_duration = np.asarray(day_duration)
axes[0].plot(day_duration_x,day_duration*1.0/max(day_duration),"k")
axes[0].plot(x_at_15,cloudiness_at_15,"k",linestyle="-")
axes[0].plot(day_duration_x,cloudiness,"r",linestyle="-")
axes[0].plot(day_duration_x,smooth_cloudiness,"b",linestyle="-")
# axes[0].twinx().plot(day_duration_x,smooth_cloudiness,"b")
illumination = day_duration*(1-smooth_cloudiness)
axes[0].twinx().plot(day_duration_x,illumination,"g")
axes[0].twinx().set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis

# # Plot each series
# axes[0].plot_date(daily_mean["Nord_268"].index, daily_mean["Nord_268"], linewidth=1,marker=" ",linestyle="-",color="k")
# axes[0].plot_date(daily_mean["EC"].index, daily_mean["EC"], linewidth=0.4,marker=" ",linestyle="-",color="r")
# axes[0].annotate(r"Reconyx Nord 268", xy=(0.55,0.20), xycoords="axes fraction",fontsize=12,color="k")
# axes[0].annotate(r"Environment Canada", xy=(0.55,0.10), xycoords="axes fraction",fontsize=12,color="r")
# Plot difference
axes[1].set_xlim(xmin=xmin,xmax=xmax)   # limit for xaxis
axes[1].plot_date(v_date2num(intersection["Nord_268"].index),intersection["Nord_268"].values - intersection["EC"].values, linewidth=0.4 ,marker=" ",linestyle="-",color="k")
axes[1].axhline(y=0, ls="-", c="k", linewidth=0.5)
# Mean diff for that year
yearly_mean_diff_ind = (intersection["Nord_268"].index > mpl.dates.num2date(xmin)) & (intersection["Nord_268"].index <= mpl.dates.num2date(xmax))
yearly_mean_diff = intersection["Nord_268"][yearly_mean_diff_ind] - intersection["EC"][yearly_mean_diff_ind]
mean_diff = np.mean(yearly_mean_diff)
axes[1].axhline(y=mean_diff, ls="--", c="b", linewidth=0.5)
axes[1].annotate(r"$\Delta T_{mean}$ = "+"{:.1f}".format(mean_diff)+r" $^\circ$C", xy=(0.04,0.75), xycoords="axes fraction",fontsize=12,color="b")
axes[1].annotate(str(year)+"-"+str(year+1), xy=(0.04,0.875), xycoords="axes fraction",fontsize=12,color="k")
axes[1].set_ylabel(r"$\Delta T_{268 - EC}$ ($^\circ$C)")
axes[1].yaxis.set_ticks(np.arange(-10,30,1), minor=True)

# print(yearly_mean_diff)
monthly_mean_diff = yearly_mean_diff.resample("m").mean()
# axes[1].plot_date(v_date2num(monthly_mean_diff.index)-15,monthly_mean_diff.values, linewidth=1 ,marker=".",linestyle="-",color="b")
smooth_diff = savgol_filter(yearly_mean_diff.values, 121, 3) # window size, polynomial order
axes[1].plot_date(yearly_mean_diff.index,smooth_diff, linewidth=1 ,marker="",linestyle="-",color="b")


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

# for i in list([1,2,4,5]):
#     axes[i].get_yaxis().set_ticklabels([])
#     axes[i].set_ylabel("")

for i in list([0]):
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].set_xlabel("")

#plt.show()
fig.savefig(figures_path[0:-3]+"png", dpi=300)
plt.close()
