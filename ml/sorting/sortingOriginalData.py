import pandas as pd
import os
import shutil
import pathlib

# orion - 1
# cassiopeia - 2
# lyra - 3
# cygnus - 4
# aquila - 5
# pleiades - 6
# taurus - 7
# canis_major - 8
# gemini - 9
# canis_minor - 10
# ursa_major - 11
# scorpius - 12
# bootes - 13
# leo - 14
# moon - 15
# sagittarius - 16

constellations = ["orion", "cassiopeia", "lyra", "cygnus", "aquila", "pleiades", "taurus", "canis_major", "gemini", "canis_minor", "ursa_major", "scorpius", "bootes", "leo", "moon", "sagittarius"]
constellationsID = {"orion":0, "cassiopeia":1, "lyra":2, "cygnus":3, "aquila":4, "pleiades":5, "taurus":6, "canis_major":7, "gemini":8, "canis_minor":9, "ursa_major":10, "scorpius":11, "bootes":12, "leo":13, "moon":14, "sagittarius":15}
constellation_count = {"orion":0, "cassiopeia":0, "lyra":0, "cygnus":0, "aquila":0, "pleiades":0, "taurus":0, "canis_major":0, "gemini":0, "canis_minor":0, "ursa_major":0, "scorpius":0, "bootes":0, "leo":0, "moon":0, "sagittarius":0}

# Get the annotations for the train data in a datframe
df = pd.read_csv("/Users/bendyson/Coding/gitRepos/mush2024/data/train/_annotations.csv")

#iterate through all the rows
for index, row in df.iterrows():
    # #Copy the file to the new location of the constellation name
    shutil.copyfile('/Users/bendyson/Coding/gitRepos/mush2024/data/train/'+row['filename'], '/Users/bendyson/Coding/gitRepos/mush2024/dataset/' + row['class'] + "/" + row['class'] + str(constellation_count[row['class']]) + pathlib.Path(row['filename']).suffix)
    # #increase the constellation number
    constellation_count[row['class']] += 1

# Get the annotations for the valid data in a datframe
df = pd.read_csv("/Users/bendyson/Coding/gitRepos/mush2024/data/valid/_annotations.csv")

#iterate through all the rows
for index, row in df.iterrows():
    #Copy the file to the new location of the constellation name
    shutil.copyfile('/Users/bendyson/Coding/gitRepos/mush2024/data/valid/'+row['filename'], '/Users/bendyson/Coding/gitRepos/mush2024/dataset/' + row['class'] + "/" + row['class'] + str(constellation_count[row['class']]) + pathlib.Path(row['filename']).suffix)
    constellation_count[row['class']] += 1




