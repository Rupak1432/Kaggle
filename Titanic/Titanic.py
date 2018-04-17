import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])

print(data_train.sample(3))

a = sns.barplot(x="Embarked" , y="Survived", hue = "Sex", data = data_train)
