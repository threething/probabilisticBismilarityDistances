import sys
import csv

from numpy import long

inputfile = sys.argv[1]
#print(inputfile)


with open(inputfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    SPItime = 0
    SPITP = 0
    SPILP = 0
    SPISETM = 0
    #VItime = 0
    #VITP = 0
    #VIIter = 0
    #error = 0
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
          SPItime += long(row[0])/1E6
          SPITP += int(row[1])
          SPILP += int(row[2])
          SPISETM += int(row[3])
        #VItime += long(row[4])/1E6
        #VITP += int(row[5])
        #VIIter += int(row[6])
        #error += float(row[7])
        line_count += 1

print(SPItime/(line_count-1), ",", SPITP/(line_count-1),",", SPILP/(line_count-1),",", SPISETM/(line_count-1))
#, ",", VItime/line_count,",", VITP/line_count,",", VIIter/line_count,",", error/line_count

