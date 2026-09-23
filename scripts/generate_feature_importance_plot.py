from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_feature_importance_plot():
  # Setup styling to match publication standards
  sns.set_theme(style='whitegrid')
  plt.rcParams.update({
      'font.size': 12,
      'axes.labelsize': 14,
      'axes.titlesize': 16,
      'xtick.labelsize': 11,
      'ytick.labelsize': 11,
      'figure.autolayout': True,
  })

  # Data from Table 2 (Logistic Regression Coefficients)
  df = pd.DataFrame({
      'Feature': [
          'Δ Shortest Path Length',
          'Global Efficiency Drop',
          'Δ Current-Flow Centrality',
          'Δ Bridging Betweenness',
      ],
      'Coefficient': [3.8072, 0.8048, 0.1457, 0.0699],
  })

  # Sort data for clean visual hierarchy
  df = df.sort_values('Coefficient', ascending=True)

  # Create figure
  fig, ax = plt.subplots(figsize=(9, 5))

  # Horizontal bar plot for readability of feature names
  bars = ax.barh(
      df['Feature'],
      df['Coefficient'],
      color='#2b5c8f',
      edgecolor='#1a365d',
      linewidth=1.2,
      height=0.55,
  )

  # Add exact value labels next to the bars
  for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.08,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.4f}',
        va='center',
        ha='left',
        fontsize=11,
        weight='bold',
        color='#2d3748',
    )

  # Formatting titles and axes
  ax.set_title(
      'Logistic Regression Feature Coefficients (Network Disruption)',
      pad=15,
      weight='bold',
  )
  ax.set_xlabel('Coefficient Weight (β)', labelpad=10)
  ax.set_xlim(0, max(df['Coefficient']) * 1.25)  # Leave room for text labels
  ax.grid(axis='y', visible=False)
  ax.grid(axis='x', linestyle='--', alpha=0.7)

  # Ensure output directory exists
  output_dir = Path('results/figures')
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / 'feature_importance_bar_plot.png'

  # Save figure
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  print(f'[*] Successfully saved feature importance plot to: {output_path}')
  plt.close()


if __name__ == '__main__':
  generate_feature_importance_plot()