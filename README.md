# heterogeneity

Hierarchical Heterogeneity across Human Cortex Shapes Large-Scale Neural Dynamics
Demirtas et al., 2019 (Neuron)

For details see the documentation at docs/h-BNM.pdf


# Kill all optimization processes
pkill -f multi_map_optim.py
pkill -f surrogate_map_optim
pkill -f "surrogate_maps/"


### Cheat Sheet 
Start the process in background and disown it
`nohup bash <script_name>.sh > logs/<script_name>.log 2>&1 & disown`

Quicly look at the logs
`tail -f <log_file>`