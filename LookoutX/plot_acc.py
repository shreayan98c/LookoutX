# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# accuracy = np.array([[25, 40, 55, 65],
#                      [20, 33.33, 40, 53.33],
#                      [13.33, 13.33, 20, 26.67]])
#
# categories = ['Easy', 'Medium', 'Hard']
# labels = ['0-shot', '1-shot', '3-shot', '5-shot']
#
# # Plotting
# fig, ax = plt.subplots()
#
# for i, cat in enumerate(categories):
#     ax.bar(labels, accuracy[i], label=cat)
#
# ax.set_xlabel('Number of Shots')
# ax.set_ylabel('Accuracy (%)')
# ax.set_title('Accuracy vs Shots')
# ax.legend()
#
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# create a dataframe with the results
results = {'Accuracy': ['Easy', 'Medium', 'Hard'],
           '0-shot': [25, 20, 13.33],
           '1-shot': [40, 33.33, 13.33],
           '3-shot': [55, 40, 20],
           '5-shot': [65, 53.33, 26.67]}

df = pd.DataFrame.from_dict(results)

# set x-axis labels
x_labels = ['Easy', 'Medium', 'Hard']

# set the figure size
plt.figure(figsize=(8, 6))

# plot the lines for each shot
plt.plot(x_labels, df['0-shot'], label='0-shot')
plt.plot(x_labels, df['1-shot'], label='1-shot')
plt.plot(x_labels, df['3-shot'], label='3-shot')
plt.plot(x_labels, df['5-shot'], label='5-shot')

# set the y-axis limits and labels
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.xlabel('Complexity of Image-Prompt pair')

# add a title and legend
plt.title('Few-shot Accuracy')
plt.legend()

# display the plot
plt.show()
