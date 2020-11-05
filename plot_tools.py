#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:26:35 2017

@author: peiji
"""

import numpy as np
import pandas as pd
import copy
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

__one_day__ = pd.to_timedelta(60*60*24,unit='s')
__four_hour__ = pd.to_timedelta(4*60*60,unit='s')

def data_bytime(data,start,end):
    """return data frame from start time to end time"""
    manipulating = data
    return manipulating.loc[manipulating.time.between(start,end)]

def data_slice_by_day(data):
    """return a list of data frame from slicing origin data frame by day"""
    #TODO
    start_time = data.time.min()
    end_time = data.time.max()
    data_slices = []
    slice_start_time = start_time.replace(hour=0,minute=0,second=0)
    while(slice_start_time < end_time):
        d = data_bytime(data,slice_start_time,slice_start_time + __one_day__)
        if(not d.empty):
            data_slices.append(d)
        slice_start_time = slice_start_time + __one_day__
    return data_slices

def plot_data(user_id,user_data,fig_path='./figs/'):

    origin = user_data[0]
    fit = user_data[1]
    #print("origin:",origin)
    begin_date = origin.measure_time[0].split(' ')[0]
    #begin_date = origin.measure_time[0]
    h1=plt.figure()
    h1.set_size_inches(10, 6)
    ax=plt.gca()
#    fig, ax = plt.subplots() 
#    plt.tight_layout()
    ax.plot_date(origin.measure_time, origin.temperature, '--b',label='origin')
    ax.plot_date(fit.measure_time, fit.temperature, 'r',label='cosinor')
    ax.legend(fontsize=12)
    ax.tick_params(direction='out', length=5, width=2, colors='k')
    xfmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ymajorLocator = MultipleLocator() #将y轴主刻度标签设置为0.2的倍数  
    ymajorFormatter = FormatStrFormatter('%1.1f') #设置y轴标签文本的格式  
    ax.yaxis.set_major_locator(ymajorLocator)  
    ax.yaxis.set_major_formatter(ymajorFormatter)  
#    ax.xaxis.set_major_formatter(xfmt)
#    fig.text(0.5, 0.96, 'ID: '+user_id+'\n'+'Begin: '+begin_date,
#             horizontalalignment='center', color='black',
#             size='large')
#    fig.text(0.5, 0.95, 'Begin:'+begin_date,
#             horizontalalignment='center', color='black',
#             size='medium')
    plt.title('user_id: '+user_id+'(healthy)'+'\n'+'Begin: '+begin_date,fontsize=12)
    plt.xlabel('time', fontsize=10)
    plt.ylabel('temperature', fontsize=10)
    ax.yaxis.grid()
    ax.xaxis.grid()
    plt.savefig(fig_path+user_id+'.pdf',dpi=300)
    plt.show()
    
    
if __name__ == "__main__":
    dt = pd.read_csv('13918950836.csv')
    dt['time'] = pd.to_datetime(dt.measure_time,unit='s')
    data_slices = data_slice_by_day(dt)
    for day in range(len(data_slices)):
        plot_data(day,data_slices[day],fig_path='./temp/')