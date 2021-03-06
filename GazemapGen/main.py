import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import collections  as mc
from pathlib import Path
from random import random
from os import mkdir, remove

# THIS DATA CAN BE SPOOFED USING ../testing/GazeGenSpoof.py
# Read in eye data
fix_data = pd.read_csv('Student_data.csv')
print("Fixation data")
print(fix_data.head())

# Read in confidence data
gaze_file = Path('../Activate/confidences.txt')
label_data = pd.read_csv(gaze_file)
print("Confidence data")
print(label_data.tail())

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
# correct_op = label_data.act_label[label_data.XDAT == xdat].values[0]
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

# Training/Test split
train_chance = 0.8  # 80% of images are training
# Check for training and test folders
if not Path("training").is_dir():
    mkdir(Path("training"))
if not Path("training/conf").is_dir():
    mkdir(Path("training/conf"))
if not Path("training/non_conf").is_dir():
    mkdir(Path("training/non_conf"))

if not Path("test").is_dir():
    mkdir(Path("test"))
if not Path("test/conf").is_dir():
    mkdir(Path("test/conf"))
if not Path("test/non_conf").is_dir():
    mkdir(Path("test/non_conf"))

timerIndex = 0
lastLabelTime = label_data.time[0]  # Get labeling start time
# For each image that was classified
for imgIndex in range(1, len(label_data)):
    currLabelTime = label_data.time[imgIndex]
    conf = label_data.Confidence[imgIndex]
    conf_string = ""
    if conf == 1:
        conf_dir = "conf"
    else:
        conf_dir = "non_conf"

    # Setup lists for lines
    mark_colors = []
    line_colors = []
    line_widths = []
    lines = []
    # set a different marker based on what the user is looking at
    markers = []

    xList = []
    yList = []
    durList = []
    interList = []
    aoiList = []
    aoi_nameList = []
    correctList = []
    sizesList = []

    # Find index of first fixation event that happened after the last label event
    startTime = fix_data.StartTime[timerIndex]
    while lastLabelTime > startTime:
        timerIndex += 1
        if timerIndex == len(fix_data.StartTime)-1:
            exit(0)
        startTime = fix_data.StartTime[timerIndex]

    stopTime = fix_data.StopTime[timerIndex]
    # Find fixations after last timer and before label happened
    # Iterate through fixations until they take place after the labelling event, add these to the gazemap
    # Check we aren't going off the end of the array too.
    while stopTime < currLabelTime and not timerIndex == len(fix_data.HorzPos)-2:
        x = fix_data.HorzPos[timerIndex]
        y = fix_data.VertPos[timerIndex]
        duration = fix_data.Duration[timerIndex]
        inter_fix = fix_data.InterfixDur[timerIndex]
        aoi = fix_data.AOI[timerIndex]
        aoi_name = fix_data.AOIName[timerIndex]
        correct_op = label_data.ActLabel[imgIndex]
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
        xplus1 = fix_data.HorzPos[timerIndex+1]
        yplus1 = fix_data.VertPos[timerIndex+1]
        lines.append([(x, y),(xplus1, yplus1)])
        aoi_nameplus1 = aoi_name = fix_data.AOIName[timerIndex+1]
        if ('Op' in aoi_name) and ('Stim' in aoi_nameplus1):
            line_colors.append('b')
        elif ('Stim' in aoi_name) and ('Op' in aoi_nameplus1):
            line_colors.append('c')
        elif ('OUT' in aoi_name) or ('OUT' in aoi_nameplus1):
            line_colors.append('k')
        else:
            line_colors.append('g')

        xList.append(x)
        yList.append(y)
        aoiList.append(aoi)
        sizesList.append(sizes)

        timerIndex += 1
        stopTime = fix_data.StopTime[timerIndex]

    if len(xList) > 1:
        fig, ax = plt.subplots(figsize=(10.0, 10.0))

        for i in range(len(aoiList)):
            ax.scatter(xList[i], yList[i], s=sizesList[i],
                       marker=markers[i], alpha=None)  # 0.5)
        lc = mc.LineCollection(lines, colors=line_colors, linewidths=line_widths, alpha=None)  # 0.5)
        ax.add_collection(lc)
        ax.set_xlim([0, 1920])
        ax.set_ylim([1080, 0])
        ax.axis('off')
        # plt.show()
        if random() > 0.8:
            fig.savefig(Path('test/' + conf_dir + "/" + str(label_data.ImageId[imgIndex]) + '.png'))
        else:
            fig.savefig(Path('training/' + conf_dir + "/" + str(label_data.ImageId[imgIndex]) + '.png'))

    lastLabelTime = currLabelTime

# Delete gaze data
remove("Student_data.csv")

