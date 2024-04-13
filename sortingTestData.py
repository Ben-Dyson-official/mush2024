import pandas as pd
import shutil
import pathlib


constellations = ["orion", "cassiopeia", "lyra", "cygnus", "aquila", "pleiades", "taurus", "canis_major", "gemini", "canis_minor", "ursa_major", "scorpius", "bootes", "leo", "moon", "sagittarius"]
constellationsID = {"orion":0, "cassiopeia":1, "lyra":2, "cygnus":3, "aquila":4, "pleiades":5, "taurus":6, "canis_major":7, "gemini":8, "canis_minor":9, "ursa_major":10, "scorpius":11, "bootes":12, "leo":13, "moon":14, "sagittarius":15}
constellation_count = {"orion":0, "cassiopeia":0, "lyra":0, "cygnus":0, "aquila":0, "pleiades":0, "taurus":0, "canis_major":0, "gemini":0, "canis_minor":0, "ursa_major":0, "scorpius":0, "bootes":0, "leo":0, "moon":0, "sagittarius":0}

testData = {"Image_id": 0, "labels": 0}
Image_id = []
labels = []

df = pd.read_csv("/Users/bendyson/Coding/gitRepos/mush2024/data/test/_annotations.csv")
#iterate through all the rows
for index, row in df.iterrows():
    #Copy the file to the new location of the constellation name
    shutil.copyfile('/Users/bendyson/Coding/gitRepos/mush2024/data/test/' + row['filename'], '/Users/bendyson/Coding/gitRepos/mush2024/dataset/test_data/' + row['class'] + "_" + str(constellation_count[row['class']]) + pathlib.Path(row['filename']).suffix)
    #increase the constellation number
    constellation_count[row['class']] += 1
    #Add the image id
    Image_id.append(row['class'] + "_" + str(constellation_count[row['class']]) + pathlib.Path(row['filename']).suffix)
    #Add the labels
    labels.append(constellationsID[row['class']])

#Add the image id and labels to testData
testData["Image_id"] = Image_id
testData["labels"] = labels

#Create the dataframe
df = pd.DataFrame(testData)

#Save the dataframe
df.to_csv('/Users/bendyson/Coding/gitRepos/mush2024/dataset/test.csv')




