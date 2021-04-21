# GazeGenSpoof.py - Creates spoof confidences.txt and StudentData.csv files
import csv
from pathlib import Path
import os
import time
import random

# Gaze data randomness
fileName = "spoof_output"
aoiName = ["Op", "Stim", "OUT"]
useVarietyOfAoi = True
startTime = curr_time = time.time_ns() * 0.001
durationMin = 0.1
durationMax = 3.0
length = 600000000  # 600s
interFixMin = 0
interFixMax = 0.5
horzMin = 0
horzMax = 1920
vertMin = 0
vertMax = 1080
aoiMin = 0
aoiMax = 5000

# Confidence randomness
imageMin = 0
imageMax = 50000
labelCount = 10
imgDivisor = round(imageMax / labelCount)
confMin = 0
confMax = 1
weightOfCorrectLabel = 90
weightOfWrongLabel = 10
minLabelTime = 1
maxLabelTime = 15


# Seconds to microseconds
def s_to_us(val):
    return val * 1000000


# Microseconds to seconds
def us_to_s(val):
    return val * 0.000001


# Setup file headers if it does not exist
if not os.path.isfile(Path('../GazemapGen/Student_data.csv')):
    with open(Path("../GazemapGen/Student_data.csv"), newline="", mode='w') as file:
        file = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file.writerow(['File', 'AOIName', 'StartTime', 'Duration', 'StopTime', 'InterfixDur', 'HorzPos', 'VertPos',
                       'AOI'])

time_count = 0
interfix = 0
# Start adding to the file
with open(Path("../GazemapGen/Student_data.csv"), newline="", mode='a') as file:
    file = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    while time_count < length:
        duration = round(random.uniform(durationMin, durationMax), 4)
        start = round(curr_time + s_to_us(interfix))
        end = round(start + s_to_us(duration))
        aoi = random.randint(aoiMin, aoiMax)
        horz = random.randint(horzMin, horzMax)
        vert = random.randint(vertMin, vertMax)
        if useVarietyOfAoi:
            aoiType = random.choice(aoiName)
        else:
            aoiType = "Stim"

        aoiCombined = aoiType + str(aoi)

        # Write data to file
        file.writerow([fileName, aoiCombined, start, duration, end, interfix, horz, vert, aoi])

        time_count += s_to_us(duration)
        interfix = random.uniform(interFixMin, interFixMax)
        curr_time = end

# Set up confidences file
if not os.path.isfile('confidences.txt'):
    with open(Path('../Activate/confidences.txt'), 'w') as thefile:
        print('ImageId,Label,Confidence,time,ActLabel,XDAT', file=thefile)
        print(
            "{}, {}, {}, {}, {}, {} ".format(0, 0, 0, curr_time, 0, 0),
            file=thefile)

# Write data
time_count = 0
with open(Path('../Activate/confidences.txt'), 'a') as thefile:
    while time_count < length:
        imgId = random.randint(imageMin, imageMax)
        labels = [imgId//imgDivisor, random.randint(0, labelCount-1)]
        user_confidence = random.uniform(confMin, confMax)
        user_label = random.choices(labels, weights=(weightOfCorrectLabel, weightOfWrongLabel), k=1)[0]

        print("{}, {}, {}, {}, {}, {} ".format(imgId, user_label, user_confidence, curr_time, labels[0], "130"),
              file=thefile)

        duration = random.uniform(minLabelTime, maxLabelTime)
        time_count += s_to_us(duration)
