import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.lines import Line2D

def objective_function(x):
    depths = np.array([0.35, 0.5, 0.6, 0.6, 0.95, 0.45])
    means = np.array([1.5, 4.0, 6.5, 9.5, 13.0, 16.5])
    widths = np.array([0.7, 0.8, 1.0, 1.0, 1.3, 0.9])
    
    y = 1.0
    for di, mui, sigmai in zip(depths, means, widths):
        y -= di * np.exp(-((x - mui)**2) / (2 * sigmai**2))
    return y

bounds = [(0, 3), (2.5, 5.5), (5, 8), (8, 11), (11.5, 14.5), (15, 18)]
solution_positions = []
for b in bounds:
    res = minimize_scalar(objective_function, bounds=b, method='bounded')
    if res.success:
        solution_positions.append((res.x, res.fun))

solution_types = ['current', 'local', 'local', 'local', 'optimal', 'local']
colors = {'current': '#f0a050', 'local': '#d04040', 'optimal': '#4060b0'}


x_curve = np.linspace(0, 20, 1000)
y_curve = objective_function(x_curve)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')



ax.set_xlim(0, 20)
ax.set_ylim(-0.1, 1.3)

ax.plot(x_curve, y_curve, color='#103060', linewidth=2.5, zorder=5)

for i, (x_pos, y_pos) in enumerate(solution_positions):
    stype = solution_types[i]
    ax.scatter(x_pos, y_pos, s=150, color=colors[stype], edgecolors='white', linewidth=1.5, zorder=10)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Solution actuelle', 
           markerfacecolor=colors['current'], markersize=12, markeredgecolor='white', markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='w', label='Minimum local', 
           markerfacecolor=colors['local'], markersize=12, markeredgecolor='white', markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='w', label='Solution optimale', 
           markerfacecolor=colors['optimal'], markersize=12, markeredgecolor='white', markeredgewidth=1.5)
]

legend = ax.legend(handles=legend_elements, loc='upper left', title="Minima locaux", 
                   frameon=False, fontsize=12)
plt.setp(legend.get_title(), fontsize=14, fontweight='bold')

ax.text(18.5, objective_function(18.5) + 0.1, "Fonction objectif", 
        fontsize=12, color='#103060', ha='center', fontweight='bold')

ax.set_axis_off()
ax.set_frame_on(False)

plt.tight_layout()
plt.show()