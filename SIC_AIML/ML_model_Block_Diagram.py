import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 10))

# Draw rectangles
rect_params = {'edgecolor': 'black', 'facecolor': 'lightgrey', 'linewidth': 2}
ax.add_patch(patches.Rectangle((0.1, 0.85), 0.8, 0.1, **rect_params))
ax.add_patch(patches.Rectangle((0.1, 0.7), 0.8, 0.1, **rect_params))
ax.add_patch(patches.Rectangle((0.1, 0.55), 0.8, 0.1, **rect_params))
ax.add_patch(patches.Rectangle((0.1, 0.4), 0.8, 0.1, **rect_params))
ax.add_patch(patches.Rectangle((0.1, 0.25), 0.8, 0.1, **rect_params))

# Add text
ax.text(0.5, 0.9, 'Load PDF Data\n(Extract Text from PDFs)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.5, 0.75, 'Preprocess Data\n(Cleaning, Tokenization)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.5, 0.6, 'Feature Extraction\n(TF-IDF)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.5, 0.45, 'Model Selection and Training\n(Random Forest, XGBoost, Naive Bayes)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.5, 0.3, 'Model Evaluation\n(Confusion Matrix, Classification Report, AUC/ROC)', horizontalalignment='center', verticalalignment='center', fontsize=12)

# Add arrows
arrow_params = {'head_width': 0.02, 'head_length': 0.02, 'fc': 'k', 'ec': 'k', 'linewidth': 2}
ax.arrow(0.5, 0.85, 0, -0.05, **arrow_params)
ax.arrow(0.5, 0.7, 0, -0.05, **arrow_params)
ax.arrow(0.5, 0.55, 0, -0.05, **arrow_params)
ax.arrow(0.5, 0.4, 0, -0.05, **arrow_params)

# Remove axes
ax.set_axis_off()

plt.show()
