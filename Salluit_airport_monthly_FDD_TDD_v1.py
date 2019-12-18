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
mpl.rcParams["font.size"] = 10. # change the size of the font in every figure
mpl.rcParams["font.family"] = "Arial" # font Arial in every figure
mpl.rcParams["font.weight"] = 100 # font Arial in every figure
mpl.rcParams["axes.labelsize"] = 10.
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
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
fig, axes = plt.subplots(3, 1, figsize=(5, 6)) # (1,1) means one plot, and figsize is w x h in inch of figure
fig.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.92, hspace=0.2, wspace=0.1) # adjust the box of axes regarding the figure size
axes = axes.flatten()
# fig.suptitle("Comparing Deception Bay Reconyx Temperature Measurements with Salluit Airport", fontsize=12)
for i, year in enumerate([2015,2016,2017]):
    xmin = mpl.dates.date2num(datetime.datetime(year=int(year),month=int(10),day=int(1)))
    xmax = mpl.dates.date2num(datetime.datetime(year=int(year)+1,month=int(9),day=int(30)))

    # Plot each series
    axes[i].set_xlim(xmin=xmin-15,xmax=xmax-15)   # limit for xaxis
    axes[i].yaxis.set_ticks(np.arange(-1000,500,400))
    axes[i].yaxis.set_ticks(np.arange(-1000,1000,200), minor=True)
    axes[i].set_ylim(-900,400)
    axes[i].annotate(str(year)+"-"+str(year+1), xy=(0.03,0.83), xycoords="axes fraction",fontsize=10,color="k")
    axes[i].axhline(y=0, ls="-", c="k", linewidth=0.5)
    if i == 0:
        axes[i].annotate("Freezing degree-days", xy=(0.60, 0.2), xycoords="axes fraction", fontsize=10,
                         color="teal")
        axes[i].annotate("Thawing degree-days", xy=(0.60, 0.08), xycoords="axes fraction", fontsize=10,
                        color="orange")
    # Cut to year
    yearly_T = daily_mean["EC"].loc[mpl.dates.num2date(xmin):mpl.dates.num2date(xmax)]
    # Check for nan; interpolate
    if yearly_T.isna().sum() > 0:
        yearly_T = yearly_T.interpolate()
    freezing_days = yearly_T.loc[yearly_T<= 0]
    thawing_days = yearly_T.loc[yearly_T > 0]
    # Group by month, sum, center at beginning of month using label
    monthly = {}
    monthly["FDD"] = freezing_days.resample("M", label="left").sum()
    monthly["TDD"] = thawing_days.resample("M", label="left").sum()

    axes[i].set_ylabel(r"CFDD and CTDD ($^\circ$C)")
    axes[i].bar(monthly["TDD"].index, monthly["TDD"], width=10, color="orange",bottom=None, align='center', data=None)
    axes[i].bar(monthly["FDD"].index, monthly["FDD"], width=10, color="teal",bottom=None, align='center', data=None)
    # for d,n in zip(monthly["TDD"].index,monthly["TDD"]):
    #     axes[i].annotate(str(n), xy=(d, n+25), xycoords="data", fontsize=10, color="orange")




    # axes[i].plot_date(daily_mean["DB"].index, daily_mean["DB"], linewidth=0.4,marker=" ",linestyle="-",color="k")
    # axes[i].plot_date(daily_mean["EC"].index, daily_mean["EC"], linewidth=0.4,marker=" ",linestyle="-",color="r")
    # if i == 0:
    #     axes[i].annotate(r"Camera", xy=(0.05,0.18), xycoords="axes fraction",fontsize=12,color="k")
    #     axes[i].annotate(r"Airport (50 km)", xy=(0.05,0.08), xycoords="axes fraction",fontsize=12,color="r")


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

    # tick_list = list()
    # for year in np.arange(2015,2019):
    #     for month in np.arange(1,13):
    #         for week_day in [8,15,22,29]:
    #             if not (month == 2 and week_day == 29):
    #                 tick_list.append(mpl.dates.date2num(datetime.datetime(year=year,month=month,day=week_day)))
    # ax.xaxis.set_minor_locator(ticker.FixedLocator(tick_list))
    ax.tick_params(direction='in',which="both",right=1,top=1)

for i in list([0,1]):
    axes[i].get_xaxis().set_ticklabels([])
    axes[i].set_xlabel("")

#plt.show()
fig.savefig(figures_path)
plt.close()
