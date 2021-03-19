import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
from matplotlib import markers as mark
import matplotlib.image as mpimg
from matplotlib import patches as patch
from matplotlib.collections import PatchCollection

# %matplotlib inline

# Read in eye data
fix_data = pd.read_csv('Student_data.csv')
print("Fixation data")
print(fix_data.head())

# Read in confidence data
summary_data = pd.read_csv('../Activate/confidences.txt')
print("Confidence data")
print(summary_data.tail())

'''
plot for 1 trial so we see what it looks like
'''
# user_id = 'test_file'
# xdat = 130
#
# x = fix_data.HorzPos[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
# y = fix_data.VertPos[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
# duration = fix_data.Duration[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
# inter_fix = fix_data.InterfixDur[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
# aoi = fix_data.AOI[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
# aoi_name = fix_data.AOIName[(fix_data.File == user_id) & (fix_data.XDAT == xdat)].values
#
# correct_op = summary_data.act_label[summary_data.XDAT == xdat].values[0]
#
# sizes = (duration + 0.1) * 80
# line_colors = []
# line_widths = []
# lines = []
# # set a different marker based on what the user is looking at
# markers = []
# for i in range(len(aoi)):
#     if (aoi[i] == correct_op) and ('Op' in aoi_name[i]):
#     if 'Op' in aoi_name[i]:
#         markers.append('*')
#     elif 'Op' in aoi_name[i]:
#         markers.append('o')
#     elif 'Stim' in aoi_name[i]:
#         markers.append('s')
#     else:
#         markers.append('x')
#     # shades of red for markers, based on pupil diam
#     # mark_colors.append((1 - pupil[i], 0, 0))  # Not used in code
#     line_widths.append((inter_fix[i] + 0.1) * 6)
#     if i < len(aoi) - 1:
#         lines.append([(x[i], y[i]), (x[i + 1], y[i + 1])])
#         if ('Op' in aoi_name[i]) and ('Stim' in aoi_name[i + 1]):
#             line_colors.append('b')
#         elif ('Stim' in aoi_name[i]) and ('Op' in aoi_name[i + 1]):
#             line_colors.append('c')
#         elif ('OUT' in aoi_name[i]) or ('OUT' in aoi_name[i + 1]):
#             line_colors.append('k')
#         else:
#             line_colors.append('g')
#
# fig, ax = plt.subplots(figsize=(10.0, 10.0))
#
# for i in range(len(aoi)):
#     ax.scatter(x[i], y[i], s=sizes[i],
#                marker=markers[i], alpha=None)  # 0.5)
# lc = mc.LineCollection(lines, colors=line_colors, linewidths=line_widths, alpha=None)  # 0.5)
# ax.add_collection(lc)
# ax.set_xlim([0, 258])
# ax.set_ylim([254, 0])
# ax.axis('off')
# plt.show()
# fig.savefig('Example' + user_id + '_' + str(xdat) + '.png')

# Code to iterate through gazemap solutions

'''
for each labelled image
{
if classification is after start and before end
{
get data and plot it?
}
}
'''
lastLabelTime = None
timerIndex = 0
# For each image that was classified
for imgId in summary_data.ImageId:
    currLabelTime = summary_data.time[(summary_data.ImageId == imgId)].values[0]
    # Get start time on first run through
    if not lastLabelTime:
        lastLabelTime = currLabelTime
        continue

    # Find index of first fixation event that happened after the last label event
    startTime = fix_data.StartTime[timerIndex]
    while lastLabelTime > startTime:
        timerIndex += 1

    # Setup lists for lines
    mark_colors = []
    line_colors = []
    line_widths = []
    lines = []
    # set a different marker based on what the user is looking at
    markers = []

    # Find fixations after last timer and before label happened
    # Iterate through fixations until they take place after the labelling event, add these to the gazemap
    while currLabelTime < fix_data.StopTime[timerIndex]:
        x = fix_data.HorzPos[timerIndex]
        y = fix_data.VertPos[timerIndex]
        duration = fix_data.Duration[timerIndex]
        inter_fix = fix_data.InterfixDur[timerIndex]
        aoi = fix_data.AOI[timerIndex]
        aoi_name = fix_data.AOIName[timerIndex]
        correct_op = summary_data.ActLabel[(summary_data.ImageId == imgId)].values[0]
        sizes = (duration + 0.1) * 80

        # Not working correctly atm
        if (aoi == correct_op) and ('Op' in aoi_name):
            markers.append('*')
        elif 'Op' in aoi_name:
            markers.append('o')
        elif 'Stim' in aoi_name:
            markers.append('s')  # This is always happening in current system
        else:
            markers.append('x')
        line_widths.append((inter_fix + 0.1) * 6)
        if i < len(aoi)-1:
            lines.append([(x[i],y[i]),(x[i+1],y[i+1])])
            if ('Op' in aoi_name[i]) and ('Stim' in aoi_name[i+1]):
                line_colors.append('b')
            elif ('Stim' in aoi_name[i]) and ('Op' in aoi_name[i+1]):
                line_colors.append('c')
            elif ('OUT' in aoi_name[i]) or ('OUT' in aoi_name[i+1]):
                line_colors.append('k')
            else:
                line_colors.append('g')

        input("wait")

input("wait")


for imgId in summary_data.ImageId:
    for i in range(len(aoi)):
        if (aoi[i] == correct_op) and ('Op' in aoi_name[i]):
            markers.append('*')
        elif 'Op' in aoi_name[i]:
            markers.append('o')
        elif 'Stim' in aoi_name[i]:
            markers.append('s')
        else:
            markers.append('x')
        #shades of red for markers, based on pupil diam
        mark_colors.append((1-pupil[i],0,0))
        line_widths.append((inter_fix[i]+0.1)*6)
        if i < len(aoi)-1:
            lines.append([(x[i],y[i]),(x[i+1],y[i+1])])
            if ('Op' in aoi_name[i]) and ('Stim' in aoi_name[i+1]):
                line_colors.append('b')
            elif ('Stim' in aoi_name[i]) and ('Op' in aoi_name[i+1]):
                line_colors.append('c')
            elif ('OUT' in aoi_name[i]) or ('OUT' in aoi_name[i+1]):
                line_colors.append('k')
            else:
                line_colors.append('g')

    fig,ax = plt.subplots(figsize = (10.0,10.0))

    for i in range(len(aoi)):
        ax.scatter(x[i],y[i],facecolors = mark_colors[i], s = sizes[i],
                marker = markers[i], alpha = None)#0.5)
    lc = mc.LineCollection(lines, colors=line_colors, linewidths=line_widths,alpha = None)#0.5)
    ax.add_collection(lc)
    ax.set_xlim([0,258])
    ax.set_ylim([254,0])
    ax.axis('off')
    plt.show()
    fig.savefig('NewImages/'+user_id+'_'+ str(xdat)+ '.png')
    plt.close('all')
