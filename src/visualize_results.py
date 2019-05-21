from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-white')

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'output' / 'results'
STL_FILE = RESULTS_DIR / 'STL.csv'

df = pd.read_csv(STL_FILE, sep=',', encoding='utf-8')

fig, ax1 = plt.subplots()
ax1.plot(df['epoch'], df['val_acc'], color='tab:orange', marker='o', label='Val. accuracy')
ax1.plot(df['epoch'], df['train_acc'], color='tab:red', marker='o', label='Train accuracy')
ax1.axhline(y=0.5, color='tab:gray', linestyle='--', label='Random')
ax1.axhline(y=0.7676, color='tab:blue', linestyle='--', label='Majority')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.tick_params('y')
ax1.set_ylim([0.4, 1])
ax1.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / (STL_FILE.stem + '_accuracy_loss_curves.png'))