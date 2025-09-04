## Dopamine 

This is just a notebook to keep track of the different receptor experiments that I ran. Hopefully this will facilitate the writing process.

### Experiment 1
In this experiment, I average all the dopamine maps, and use the resulting map to constrain the model.

### Experiment 2
In this experiment, I attempt to separate the different receptors into their respective pathways. Based on a *short* search of the litterature, I see two general categories of Dopamine receptors:
- D1-like (DRD1 & DRD5): Predominantly excitatory, central to direct pathway, mesocortical processing, higher cognition.
- D2-like (DRD2/3/4): Mostly inhibitory, key in indirect pathway, reward modulation, and presynaptic regulation.

Considering that D1 and D2 have opposing effects physiologically [1], D1-like receptors are are excitatory, while D2-like receptors are inhibitory. As a sanity check, I should run all 4 permutations, and it would be a validation of my model to see that the model fits better to empirical data.

==****So make sure to rename the first D2 run because it's inverted!!!!****===

### References
[1] Paul, M. L., Graybiel, A. M., David, J. C., & Robertson, H. A. (1992). D1-like and D2-like dopamine receptors synergistically activate rotation and c-fos expression in the dopamine-depleted striatum in a rat model of Parkinson’s disease. Journal of Neuroscience, 12(10), 3729–3742. https://doi.org/10.1523/JNEUROSCI.12-10-03729.1992
