import sys
import csv
import os

from utils.constant import fileResult, file
inputfile = file
#print(inputfile)


with open(inputfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    SPItime = 0
    SPITP = 0
    SPILP = 0
    VItime = 0
    VITP = 0
    VIIter = 0
    error = 0
    line_count = 0
    for row in csv_reader:
        SPItime += int(row[0])/1E6
        SPITP += int(row[1])
        SPILP += int(row[2])
        VItime += int(row[3])/1E6
        VITP += int(row[4])
        VIIter += int(row[5])
        error += float(row[6])
        line_count += 1

print(SPItime/line_count , "," , SPITP/line_count , "," , SPILP/line_count , "," , VItime/line_count , "," , VITP/line_count , "," , VIIter/line_count , "," , error/line_count)
