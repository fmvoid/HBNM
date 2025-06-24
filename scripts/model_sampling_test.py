import numpy as np
import os
from hbnm.io import Data
from hbnm.bnm import Bnm
from hbnm.model.utils import subdiag, fisher_z
from scipy.spatial.distance import squareform


### TODO maybe move these two plotting functions to the utils ###
def matrix_plot(ax, x, cmap, add_colorbar=True, n_ticks=5):
    im = ax.pcolormesh(x, cmap=cmap, vmin=x.min(), vmax=x.max())
    ax.set_aspect(1)
    ax.set_xlim([0, x.shape[1]])
    ax.set_ylim([0, x.shape[0]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        # Create evenly spaced ticks from min to max
        ticks = np.linspace(x.min(), x.max(), n_ticks)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])
    
    return im

def reg_plot(ax, x, y):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.scatter(x, y)
    ax.set_xlabel('model FC - z-transformed')
    ax.set_ylabel('empirical FC - z-transformed')
    text = 'r = ' + '{:3.2f}'.format(np.corrcoef(x, y)[0,1])
    ax.text(0.5, 0.95, text, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes)


# set input and output directories
current_path = os.getcwd()
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
input_dir = current_path + '/data/'
output_dir = current_path + '/outputs/'

# load the empirical data from the demirtas folder
data = Data(input_dir, output_dir)
sc, hmap, fc_obj = data.load_demirtas_data() # this function may have a confusing name...  

# load the iteration data from the output folder
fin = data.load('heterogeneous/iteration_2.hdf5', from_output=True)
theta = fin['theta'][:]
fin.close()

# set model samples
heterogeneous = Bnm(sc, gradient=hmap)
heterogeneous.set('w_EI', (theta[0,0], theta[1,0]))
heterogeneous.set('w_EE', (theta[2,0], theta[3,0]))
heterogeneous.set('G', theta[4,0])
heterogeneous.moments_method()

from scipy.stats import pearsonr
print(pearsonr(subdiag(fc_obj), subdiag(heterogeneous.get('corr_bold'))))

# Plots
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['axes.labelsize'] = 26.0
mpl.rcParams['axes.titlesize'] = 26.0
mpl.rcParams['xtick.labelsize'] = 20.0
mpl.rcParams['ytick.labelsize'] = 20.0

fig, axes = plt.subplots(1,3,figsize=(30, 10))

#import pdb; pdb.set_trace()
matrix_plot(axes[0], fc_obj, 'RdBu_r')
axes[0].set_title('Empirical FC')
matrix_plot(axes[1], heterogeneous.get('corr_bold'), 'RdBu_r')
axes[1].set_title('Heterogeneous model FC')

reg_plot(axes[2], fisher_z(subdiag(heterogeneous.get('corr_bold'))), fisher_z(subdiag(fc_obj)))
axes[2].set_title('Heterogeneous model FC fit')

plt.tight_layout()

plt.savefig(f'{output_dir}/Example_model_fit_mine_2.png', dpi=300)
plt.show()