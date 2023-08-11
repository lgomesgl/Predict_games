import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import main_data

data = main_data()

plt.subplot(2, 2, 1)
sns.barplot(data=data, x='Home', y='Home_points')
plt.xticks(rotation=90)

plt.subplot(2, 2, 2)
sns.barplot(data=data, x='Home', y='Away_points')
plt.xticks(rotation=90)

plt.subplot(2, 2, 3)
sns.barplot(data=data, x='Away', y='Home_points')
plt.xticks(rotation=90)
  
plt.subplot(2, 2, 4)
sns.barplot(data=data, x='Away', y='Away_points')
plt.xticks(rotation=90)
# plt.show()

plt.subplot(1, 2, 1)
sns.barplot(data=data, x='Day', y='Points', palette='mako')
plt.title('Points x Day of week')

plt.subplot(1, 2, 2)
sns.barplot(data=data, x='Month', y='Points', palette='mako')
plt.title('Points x Month')
plt.show()

sns.histplot(data=data, x = data['Points'], bins=30, kde=True)
plt.title('Distribuiton of Points')
plt.show()