import warnings
import os
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread


warnings.filterwarnings('ignore')

# Get all the paths
data_dir_list = os.listdir('/Users/bendyson/Coding/gitRepos/mush2024/dataset')

path, dirs, files = next(os.walk('/Users/bendyson/Coding/gitRepos/mush2024/dataset'))
file_count = len(files)

# Make a new base directory
orginal_dataset_dir = '/Users/bendyson/Coding/gitRepos/mush2024/dataset'
base_dir = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data'
# No longer used as already created
# os.mkdir(base_dir)

# Create two folders - one for train and one for validation
train_dir = os.path.join(base_dir, 'train')
# No longer used as already created
# os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
# No longer used as already created
# os.mkdir(validation_dir)

# In train create 16 folders for each constellation we are training for
# ('pleiades', 'taurus', 'sagittarius', 'gemini', 'canis_major', 'cygnus',
# 'moon', 'orion', 'cassiopeia', 'scorpius', 'aquila', 'bootes', 'leo', 'canis_minor', 'ursa_major', 'lyra')

train_pleiades_dir = os.path.join(train_dir, 'pleiades')
# No longer used as already created
# os.mkdir(train_pleiades_dir)

train_taurus_dir = os.path.join(train_dir, 'taurus')
# No longer used as already created
# os.mkdir(train_taurus_dir)

train_sagittarius_dir = os.path.join(train_dir, 'sagittarius')
# No longer used as already created
# os.mkdir(train_sagittarius_dir)

train_gemini_dir = os.path.join(train_dir, 'gemini')
# No longer used as already created
# os.mkdir(train_gemini_dir)

train_canis_major_dir = os.path.join(train_dir, 'canis_major')
# No longer used as already created
# os.mkdir(train_canis_major_dir)

train_cygnus_dir = os.path.join(train_dir, 'cygnus')
# No longer used as already created
# os.mkdir(train_cygnus_dir)

train_moon_dir = os.path.join(train_dir, 'moon')
# No longer used as already created
# os.mkdir(train_moon_dir)

train_orion_dir = os.path.join(train_dir, 'orion')
# No longer used as already created
# os.mkdir(train_orion_dir)

train_cassiopeia_dir = os.path.join(train_dir, 'cassiopeia')
# No longer used as already created
# os.mkdir(train_cassiopeia_dir)

train_scorpius_dir = os.path.join(train_dir, 'scorpius')
# No longer used as already created
# os.mkdir(train_scorpius_dir)

train_aquila_dir = os.path.join(train_dir, 'aquila')
# No longer used as already created
# os.mkdir(train_aquila_dir)

train_bootes_dir = os.path.join(train_dir, 'bootes')
# No longer used as already created
# os.mkdir(train_bootes_dir)

train_leo_dir = os.path.join(train_dir, 'leo')
# No longer used as already created
# os.mkdir(train_leo_dir)

train_canis_minor_dir = os.path.join(train_dir, 'canis_minor')
# No longer used as already created
# os.mkdir(train_canis_minor_dir)

train_ursa_major_dir = os.path.join(train_dir, 'ursa_major')
# No longer used as already created
# os.mkdir(train_ursa_major_dir)

train_lyra_dir = os.path.join(train_dir, 'lyra')
# No longer used as already created
# os.mkdir(train_lyra_dir)

# In validation create 16 folders for each constellation we are training for
# ('pleiades', 'taurus', 'sagittarius', 'gemini', 'canis_major', 'cygnus',
# 'moon', 'orion', 'cassiopeia', 'scorpius', 'aquila', 'bootes', 'leo', 'canis_minor', 'ursa_major', 'lyra')

validation_pleiades_dir = os.path.join(validation_dir, 'pleiades')
# No longer used as already created
# os.mkdir(validation_pleiades_dir)

validation_taurus_dir = os.path.join(validation_dir, 'taurus')
# No longer used as already created
# os.mkdir(validation_taurus_dir)

validation_sagittarius_dir = os.path.join(validation_dir, 'sagittarius')
# No longer used as already created
# os.mkdir(validation_sagittarius_dir)

validation_gemini_dir = os.path.join(validation_dir, 'gemini')
# No longer used as already created
# os.mkdir(validation_gemini_dir)

validation_canis_major_dir = os.path.join(validation_dir, 'canis_major')
# No longer used as already created
# os.mkdir(validation_canis_major_dir)

validation_cygnus_dir = os.path.join(validation_dir, 'cygnus')
# No longer used as already created
# os.mkdir(validation_cygnus_dir)

validation_moon_dir = os.path.join(validation_dir, 'moon')
# No longer used as already created
# os.mkdir(validation_moon_dir)

validation_orion_dir = os.path.join(validation_dir, 'orion')
# No longer used as already created
# os.mkdir(validation_orion_dir)

validation_cassiopeia_dir = os.path.join(validation_dir, 'cassiopeia')
# No longer used as already created
# os.mkdir(validation_cassiopeia_dir)

validation_scorpius_dir = os.path.join(validation_dir, 'scorpius')
# No longer used as already created
# os.mkdir(validation_scorpius_dir)

validation_aquila_dir = os.path.join(validation_dir, 'aquila')
# No longer used as already created
# os.mkdir(validation_aquila_dir)

validation_bootes_dir = os.path.join(validation_dir, 'bootes')
# No longer used as already created
# os.mkdir(validation_bootes_dir)

validation_leo_dir = os.path.join(validation_dir, 'leo')
# No longer used as already created
# os.mkdir(validation_leo_dir)

validation_canis_minor_dir = os.path.join(validation_dir, 'canis_minor')
# No longer used as already created
# os.mkdir(validation_canis_minor_dir)

validation_ursa_major_dir = os.path.join(validation_dir, 'ursa_major')
# No longer used as already created
# os.mkdir(validation_ursa_major_dir)

validation_lyra_dir = os.path.join(validation_dir, 'lyra')
# No longer used as already created
# os.mkdir(validation_lyra_dir)

def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    valid_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in valid_set:
        this_file = SOURCE + filename
        destination = VALIDATION + filename
        copyfile(this_file, destination)

AQULIA_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/aquila/'
TRAINING_AQULIA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/aquila/'
VALID_AQULIA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/aquila/'

BOOTES_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/bootes/'
TRAINING_BOOTES_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/bootes/'
VALID_BOOTES_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/bootes/'

CANIS_MAJOR_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/canis_major/'
TRAINING_CANIS_MAJOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/canis_major/'
VALID_CANIS_MAJOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/canis_major/'

CANIS_MINOR_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/canis_minor/'
TRAINING_CANIS_MINOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/canis_minor/'
VALID_CANIS_MINOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/canis_minor/'

CASSIOPEIA_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/cassiopeia/'
TRAINING_CASSIOPEIA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/cassiopeia/'
VALID_CASSIOPEIA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/cassiopeia/'

CYGNUS_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/cygnus/'
TRAINING_CYGNUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/cygnus/'
VALID_CYGNUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/cygnus/'

GEMINI_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/gemini/'
TRAINING_GEMINI_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/gemini/'
VALID_GEMINI_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/gemini/'

LEO_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/leo/'
TRAINING_LEO_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/leo/'
VALID_LEO_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/leo/'

LYRA_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/lyra/'
TRAINING_LYRA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/lyra/'
VALID_LYRA_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/lyra/'

MOON_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/moon/'
TRAINING_MOON_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/moon/'
VALID_MOON_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/moon/'

ORION_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/orion/'
TRAINING_ORION_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/orion/'
VALID_ORION_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/orion/'

PLEIADES_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/pleiades/'
TRAINING_PLEIADES_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/pleiades/'
VALID_PLEIADES_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/pleiades/'

SAGITTARIUS_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/sagittarius/'
TRAINING_SAGITTARIUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/sagittarius/'
VALID_SAGITTARIUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/sagittarius/'

SCORPIUS_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/scorpius/'
TRAINING_SCORPIUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/scorpius/'
VALID_SCORPIUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/scorpius/'

TAURUS_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/taurus/'
TRAINING_TAURUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/taurus/'
VALID_TAURUS_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/taurus/'

URSA_MAJOR_SOURCE_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/dataset/ursa_major/'
TRAINING_URSA_MAJOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/ursa_major/'
VALID_URSA_MAJOR_DIR = '/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/ursa_major/'

# How much of the data is going to train and to validate
split_size = .85


# split_data(AQULIA_SOURCE_DIR, TRAINING_AQULIA_DIR, VALID_AQULIA_DIR, split_size)
# split_data(BOOTES_SOURCE_DIR, TRAINING_BOOTES_DIR, VALID_BOOTES_DIR, split_size)
# split_data(CANIS_MAJOR_SOURCE_DIR, TRAINING_CANIS_MAJOR_DIR, VALID_CANIS_MAJOR_DIR, split_size)
# split_data(CANIS_MINOR_SOURCE_DIR, TRAINING_CANIS_MINOR_DIR, VALID_CANIS_MINOR_DIR, split_size)
# split_data(CASSIOPEIA_SOURCE_DIR, TRAINING_CASSIOPEIA_DIR, VALID_CASSIOPEIA_DIR, split_size)
# split_data(CYGNUS_SOURCE_DIR, TRAINING_CYGNUS_DIR, VALID_CYGNUS_DIR, split_size)
# split_data(GEMINI_SOURCE_DIR, TRAINING_GEMINI_DIR, VALID_GEMINI_DIR, split_size)
# split_data(LEO_SOURCE_DIR, TRAINING_LEO_DIR, VALID_LEO_DIR, split_size)
# split_data(LYRA_SOURCE_DIR, TRAINING_LYRA_DIR, VALID_LYRA_DIR, split_size)
# split_data(MOON_SOURCE_DIR, TRAINING_MOON_DIR, VALID_MOON_DIR, split_size)
# split_data(ORION_SOURCE_DIR, TRAINING_ORION_DIR, VALID_ORION_DIR, split_size)
# split_data(PLEIADES_SOURCE_DIR, TRAINING_PLEIADES_DIR, VALID_PLEIADES_DIR, split_size)
# split_data(SAGITTARIUS_SOURCE_DIR, TRAINING_SAGITTARIUS_DIR, VALID_SAGITTARIUS_DIR, split_size)
# split_data(SCORPIUS_SOURCE_DIR, TRAINING_SCORPIUS_DIR, VALID_SCORPIUS_DIR, split_size)
# split_data(TAURUS_SOURCE_DIR, TRAINING_TAURUS_DIR, VALID_TAURUS_DIR, split_size)
# split_data(URSA_MAJOR_SOURCE_DIR, TRAINING_URSA_MAJOR_DIR, VALID_URSA_MAJOR_DIR, split_size)


# Visulize the dat split
image_folder = ["orion", "cassiopeia", "lyra", "cygnus", "aquila", "pleiades", "taurus", "canis_major", "gemini", "canis_minor", "ursa_major", "scorpius", "bootes", "leo", "moon", "sagittarius"]

constellations = ["orion", "cassiopeia", "lyra", "cygnus", "aquila", "pleiades", "taurus", "canis_major", "gemini", "canis_minor", "ursa_major", "scorpius", "bootes", "leo", "moon", "sagittarius"]
constellationsID = {"orion":0, "cassiopeia":1, "lyra":2, "cygnus":3, "aquila":4, "pleiades":5, "taurus":6, "canis_major":7, "gemini":8, "canis_minor":9, "ursa_major":10, "scorpius":11, "bootes":12, "leo":13, "moon":14, "sagittarius":15}
constellation_count = {"orion":0, "cassiopeia":0, "lyra":0, "cygnus":0, "aquila":0, "pleiades":0, "taurus":0, "canis_major":0, "gemini":0, "canis_minor":0, "ursa_major":0, "scorpius":0, "bootes":0, "leo":0, "moon":0, "sagittarius":0}

nimgs = {}

for i in image_folder:
    nimages = len(os.listdir('/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/train/' + i + '/'))
    nimgs[i] = nimages
print(nimgs)
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Training Dataset')
# plt.show()

for i in image_folder:
    nimages = len(os.listdir('/Users/bendyson/Coding/gitRepos/mush2024/constellation_data/validation/'+i+'/'))
    nimgs[i]=nimages
plt.figure(figsize=(9, 6))
plt.bar(range(len(nimgs)), list(nimgs.values()), align='center')
plt.xticks(range(len(nimgs)), list(nimgs.keys()))
plt.title('Distribution of different classes in Validation Dataset')
# plt.show()


