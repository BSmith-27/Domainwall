import distributed_svpg

'''
Distributed SVPG
------
By P. Sao, R. Kannan, R. Vasudevan
Here we implement a distributed version of Stein variational policy gradient 
(see https://arxiv.org/abs/1704.02399 for details on the algorithm).

The above paper does not discuss a parallell implementation. Here, we parallelize by 
placing agents on GPUs, simulations on CPUs and run M simulations with N agents in parallel
The algorithm at this point is synchronous, but we plan to also test asynchronous versions. 

Note that the number of simulations per agent update is batch_size*numSimRuns
And number of agents is number of processes / batch_size
For fastest performance you will want numSimRuns to be 1. But it requires more cores.
Finally, for memory consider that we have number of neural nets to be equal to number of processes
By default we only place 6 NNs per GPU. This is doable for small networks, not for larger ones.
'''

#TODO: Dynamically adjust the number of agents per GPU based on network size and batch update size.
save_folder_name = 'a2c_wallenv_results/'
import os
if not os.path.exists(save_folder_name):
    try: 
        os.mkdir(save_folder_name)
        print('Making directory {}'.format(save_folder_name))
    except: 
        pass
    
distributed_svpg.train_svpg_MPI(iterations = 1000, batch_size = 4, numSimRuns=4, svpg = True,
                                gamma = 0.90, stein_temp = 10.0, save_folder_name = save_folder_name)

