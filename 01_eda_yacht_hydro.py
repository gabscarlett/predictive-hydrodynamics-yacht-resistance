"""
Baseline data investigation and correlation analysis.

This script performs a preliminary investigation into the Yacht Hydrodynamics 
Dataset to establish baseline linear relationships between hull geometry, 
Froude number, and residuary resistance.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_yacht_data(filepath):
    """Load and format yacht hydrodynamic data from CSV."""
    col_names = ['Centre_buoyancy', 'Prismatic_coeff', 'Len_displacement_ratio', 
                 'Beam_draught_ratio', 'Len_beam_ratio', 'Froude_number', 'Resistance']
    data = pd.read_csv(filepath)
    data.columns = col_names
    return data


if __name__ == "__main__":
    # Load data
    df = load_yacht_data('yacht_hydro.csv')
    corr_mat = df.corr()
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title('Correlation Coefficient Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('assets/correlation_martix.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # Plot 2: Geometry Sensitivity Bar Chart
    geometry_sensitivity = corr_mat['Resistance'].drop(['Resistance', 'Froude_number']).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    geometry_sensitivity.plot(kind='bar', color='teal')
    plt.title('Resistance Sensitivity Analysis (Geometry Only)')
    plt.ylabel('Correlation Coefficient')
    #plt.xlabel('Hull Parameters')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('assets/geometry_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # Plot 3: Froude Number vs Resistance Scatter - FIXED
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get current axes
    df.plot(kind='scatter', x='Froude_number', y='Resistance', ax=ax)  # Pass ax explicitly
    plt.title('Resistance vs Froude Number')
    plt.xlabel('Froude Number')
    plt.ylabel('Resistance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('froude_vs_resistance.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    
    # Keep plots open
    input("\nPress Enter to close all plots...")
