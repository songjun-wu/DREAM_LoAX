import os
import shutil
import numpy as np
from scipy.stats import uniform

from DREAM_LoAX.core import run_dream
from DREAM_LoAX.parameters import SampledParam
from scipy.stats import multivariate_normal

def setAttributes():
    attributes = {
        'model_name'            : 'gaussian_test',         # the name of model   

        'N'                     : 50,    # dimensionality of problem


        'nchains'               : 50,       # number of chains
        'niterations'           : 100000,   # number of maximum iterations
        'save_limits_interval'  : 100,      # The interval to save limits
        'adjust_limits_nsample' : 100,      # The interval to adjust limits
        'param_DREAM_LoAX'       : [ 0.1,    #likeli_too_bad - adjust_limits
                                    0.5,    #likeli_too_good - adjust_limits
                                    0.01,   # b - adjust_limits
                                    1.5,    # c - adjust_weights
                                    25,     # m - metrop_select
                                    0.002 * 0.01   # n - metrop_select
                                        ],

        'x_sample_number'       : 1000,     # The number of points are generated as reference for gaussian calibration


    }

    attributes['inputPath'] = '/home/wusongj/dmc_plot/'      # path for inputs
    attributes['scratchPath'] = r"/data/scratch/wusongj/test/" + attributes['model_name'] + '/'
    attributes['savePath'] = attributes['scratchPath']+'/results/'      # path for outpus
    attributes['runPath'] = attributes['scratchPath']+ '/run/'          # path for model execution


    attributes['burn_in_period'] = int(attributes['niterations'] * 0.5)        # The timesteps for burn-in

    attributes['param_min'] = np.append(np.full(attributes['N'], -2), np.full(attributes['N'], 0))          # The minimum of parameters
    attributes['param_max'] = np.append(np.full(attributes['N'],  2), np.full(attributes['N'], attributes['N']*1.5))    # The maximum of parameters

    return attributes

def createDirectory(path, replace):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)

def sortDirectory(scratchPath, replace):
    createDirectory(scratchPath + '/run/', replace)
    createDirectory(scratchPath + '/results/', replace)
    createDirectory(scratchPath + '/post/', replace)
    createDirectory(scratchPath + 'plots/', replace)

# The functions to calculate likelihood
def likelihood_gaussian(param, chainID, limits, weights, obs_all):  

    attributes = setAttributes()
    scratchPath = attributes['scratchPath']

    x = np.loadtxt(scratchPath + 'x.txt')

    y = calculation_gaussian(param, x)

    limits = limits.reshape(2, -1)
    limits_lower = limits[0,:]
    limits_upper = limits[1,:]

    likeli_lower = (y - limits_lower) / (obs_all - limits_lower)
    likeli_upper = (y - limits_upper) / (obs_all - limits_upper)

    likeli_lower[likeli_lower > 1] = 1
    likeli_lower[likeli_lower < 0] = 0
    likeli_upper[likeli_upper > 1] = 1
    likeli_upper[likeli_upper < 0] = 0

    likeli = np.sum(likeli_lower * likeli_upper * weights)

    #likeli_lower =  y > limits_lower
    #likeli_upper =  y < limits_upper
    #likeli = np.sum(np.logical_and(y < limits_upper, y > limits_lower) * weights)


    return likeli, likeli_lower, likeli_upper

def calculation_gaussian(param, x):
    attributes = setAttributes()
    N = attributes['N']
    param_min = attributes['param_min'].reshape(2,-1)
    param_max = attributes['param_max'].reshape(2,-1)

    param = param.reshape(2,-1)
    mu = param_min[0,:] + (param_max[0,:] - param_min[0,:]) * param[0,:]
    sigma = np.array([])
    for i in range(N):
        tmp = np.full(N, 0.0)
        tmp[i] = param_min[1,i] + (param_max[1,i] - param_min[1,i]) * param[1,i]
        sigma = np.append(sigma, tmp)
    sigma = sigma.reshape(N,N)


    try:
        var = multivariate_normal(mean=mu, cov=sigma)
        y = var.pdf(x)
    except:
        y = np.full(x.shape[0], 0)
        print('wrong!')
    #print(y)
    return y


# The function to define limits. Can be defined based on the modelling purpose
def define_LoA(y, nchains, uncertainty_in_percent):

    y_range = np.max(y) - np.min(y)
    limits_upper = y + uncertainty_in_percent * y_range
    limits_lower = y - uncertainty_in_percent * y_range
    limits_lower[limits_lower<0] = 0
    
    limits = np.append(limits_lower, limits_upper)
    limits_all = [limits] * nchains


    return limits_all




def main():
    # Set attributes in function setAttributes. See description in function.
    attributes = setAttributes()
    model_name = attributes['model_name']      
    scratchPath = attributes['scratchPath']
    savePath = attributes['savePath']
    N = attributes['N']
    nchains = attributes['nchains']
    niterations = attributes['niterations']
    save_limits_interval = attributes['save_limits_interval']
    adjust_limits_nsample = attributes['adjust_limits_nsample']
    x_sample_number = attributes['x_sample_number']
    burn_in_period = attributes['burn_in_period']

    batch_id = 0
    total_iterations = niterations

    # Create directiories
    sortDirectory(scratchPath, replace=False)

    # Generate reference points for calibration

    mu = np.full(N, 0)
    sigma = np.array([])
    for i in range(N):
        tmp = np.full(N, 0.0)
        tmp[i] = i + 1
        sigma = np.append(sigma, tmp)
    sigma = sigma.reshape(N,N)
    param_min = attributes['param_min'].reshape(2,-1)
    param_max = attributes['param_max'].reshape(2,-1)
    part1 = (mu - param_min[0,:]) / (param_max[0,:] - param_min[0,:])
    part2 = (np.arange(1, N+1, 1) - param_min[1,:]) / (param_max[1,:] - param_min[1,:])
    param = np.append(part1, part2)
 
    x = (np.random.random(size=x_sample_number * N) - 0.5) * 4
   
    x = x.reshape(-1, N)
    y = np.array([])
    for i in range(x_sample_number):
        y = np.append(y, calculation_gaussian(param, x[i,:]))
    np.savetxt(scratchPath + 'param.txt', param)
    np.savetxt(scratchPath + 'x.txt', x)
    np.savetxt(scratchPath + 'y.txt', y)




    # Define the limits 
    initial_limits = define_LoA(y, nchains, uncertainty_in_percent=0.25)                 # define the initial limits
    hard_limits_small = define_LoA(y, nchains, uncertainty_in_percent=0.001)    # define the lower boundary of limits
    hard_limits_large = define_LoA(y, nchains, uncertainty_in_percent=0.5)      # define the upper boundary of limits

    weights = [np.full(x_sample_number, 1/x_sample_number)]     # set the initial weights for observation points

    parameters_to_sample = SampledParam(uniform, loc=np.full(N * 2, 0.0), scale=1)  # parameters are priorly asssumed to be uniformly distributed



    run_dream(  savePath=savePath,          # path to save outputs
                parameters=[parameters_to_sample],  # parameter distribution
                likelihood=likelihood_gaussian,     # the function for likelihood calculation. See details in def likelihood_gaussian
                niterations=niterations,            # The total iterations
                total_iterations=total_iterations,  # This iteration will be used as suffix of filename of all outputs
                nchains=nchains,                    # Number of chains
                multitry=False,             
                gamma_levels=4,
                adapt_gamma=True,
                history_thin=10,                    # The interval to save DREAM chain status, and for screen outputs
                model_name=model_name,              # The name of model
                restart=False,                      # If this is a restart execution?
                limits=initial_limits,              # The inital boundary of limits
                weights=weights,                    # The initial weights of observation points
                obs_all=y,                          # The array of observations
                save_limits_interval=save_limits_interval,      # The interval to save limits
                adjust_limits_nsample=adjust_limits_nsample,    # The interval to adjust limits
                hard_limits_large=hard_limits_large,            # The upper boundayr of limits
                hard_limits_small=hard_limits_small,            # The lower boundayr of limits
                burn_in_period=burn_in_period,                  # The burn-in period (in iteration)
                param_DREAM_LoA=attributes['param_DREAM_LoAX']   # Hyper-parameters for DREAM(LoAX)
                    )



main()


