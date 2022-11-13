import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

presumptions = [0.993, 0.983, 0.971, 0.871, 0.885, 0.867, 0.839, 0.752, 0.762, 0.470, 0.437, 0.471]

general = [0.996, 0.989, 0.98, 0.967, 0.953, 0.944, 0.838, 0.791, 0.754, 0.519, 0.524, 0.487]

data = [presumptions, general]

fig1, ax1 = plt.subplots()
ax1.set_title("Performance of models trained with Presuption-enabled specificity and general specificity")
ax1.boxplot(data)
bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax1.set(axisbelow=True, title="Performance of models trained with Presuption-enabled specificity and general specificity", ylabel="F1 Score")
box_colors = ['darkkhaki', 'royalblue']
num_boxes = len(data)
medians = np.empty(num_boxes)
for i in range(num_boxes):
    box = bp['boxes'][i]
    box_x = []
    box_y = []
    for j in range(5):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    # Alternate between Dark Khaki and Royal Blue
    ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    median_x = []
    median_y = []
    for j in range(2):
        median_x.append(med.get_xdata()[j])
        median_y.append(med.get_ydata()[j])
        ax1.plot(median_x, median_y, 'k')
    medians[i] = median_y[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
             color='w', marker='*', markeredgecolor='k')

ax1.set_xticklabels(["Presupmtion-enabled Specificity", "General Specificity"], fontsize=8)