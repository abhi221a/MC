import numpy as np
import math
import random
from matplotlib import pyplot as plt

PI =3.1415926
e=2.71828

#defining some helper functions 
def get_rand_number(min_value, max_value):
    """
    This function gets a random number from a uniform distribution
    between the two input values [min_values, max_values] inclusively
    Args:
    - min_value (float)
    - max_values (float)

    Return:
    - Random nuimber between this range (float)

    """
    range = max_value - min_value
    choice = random.uniform(0,1)
    return min_value + range*choice

#function to integrate
def f_of_x(x):
    """
    This is the main function we want5 to integrate over.
    Args: 
    - x (float) :input to funtion; must be in radians
    Return :
    -Output of function f(x) (float)


    """
    return (e**(-1*x))/(1+(x-1)**2)

def crude_monte_carlo(num_samples=5000):
    """
    This function peforms the crude monte Carlo for our 
    specific function f(x) on the range x=0 to x=5.
    Notice that this bound is sufficient because f(x)
    approaches 0 at around PI
    Args:
     -num_samples (float) : numbers of samples
     Return:
     -Crude Monte Car;p estimation (float)

     """
    lower_bound = 0
    upper_bound = 5
    
    sum_of_samples = 0
    for i in range(num_samples):
        x = get_rand_number(lower_bound, upper_bound)
        sum_of_samples += f_of_x(x)
    return (upper_bound - lower_bound) * float(sum_of_samples/num_samples)
def get_crude_MC_variance(num_samples):
    """

    This function returns the variance of the Crude Monte Carlo.
    Note that he inputed number of the samples does not neccessarily
    need to correnspond to number os the samples used in the MOnte CarloSimulation.
    Args:
    - num_ samples  (int)
    Retun:
    - variance for Ceude MOnte Carlo approximation of f(x) (float)


    """
    int_max = 5 # this is the max of our interation rangw

    #get the averagw of squares

    running_total = 0

    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x)**2
    sum_of_sqs = running_total*int_max / num_samples

    # get square of average 
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total = f_of_x(x)
    sq_ave = (int_max*running_total/num_samples)**2
    return sum_of_sqs - sq_ave

    # Now we will run a Crude Monte Carlo simulation with 10000 samples 
    # We will also calculate the varience with 10000 samples and the error

MC_samples = 10000
var_samples = 10000 # number of samples we will use to calculate the varience

crude_estimation = crude_monte_carlo(MC_samples)
variance = get_crude_MC_variance(var_samples)
error = math.sqrt(variance/MC_samples)


#display results

print(f"Monte Carlo Approximation of f(x):{crude_estimation}")
print(f"Variance of Approximation:{variance}")
print(f"Error in Approximation: {error}")




#IMPORTANCE SAMPLING

# Determine the Optimal Weight Function Template

#plot the function

xs1 = [float(i/50) for i in range(int(50*PI*2))]
ys1 = [f_of_x(x) for x in xs1]
plt.plot(xs1,ys1)
plt.title("f(x)")
plt.show()
# this is the template of our weight function g(x)

def g_of_x(x, A, lamda):
    e = 2.71828
    return A*math.pow(e, -1*lamda*x)

xs2 = [float (i/50) for i in range(int(50*PI))]
fs2 = [f_of_x(x) for x in xs2]
gs2 = [g_of_x(x, A=1.4, lamda=1.4) for x in xs2]
plt.plot(xs2, fs2)
plt.plot(xs2, gs2)
plt.title("f(x) and g(x)")
plt.show()

#Determine the Optimal Parameters for our Weight Function

def inverse_G_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda

def get_IS_variance(lamda, num_samples):
    """
    This function calculates the variance if a monte carlo using importance sampling .
    arg:
    - lamda (float): lambda value of g(x) being tested
    Return :
    - Variance
    """
    A = lamda
    int_max = 5

    # get sum of squares

    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += (f_of_x(x)/g_of_x(x, A, lamda))**2
    sum_of_sqs = running_total / num_samples

    #get squared average
    running_total = 0
    for i in range(num_samples):
        x = get_rand_number(0, int_max)
        running_total += f_of_x(x)/g_of_x(x, A, lamda)
    sq_ave = (running_total/num_samples)**2

    
    return sum_of_sqs - sq_ave


# get variance as a function of lamda by testing many different lamda


test_lamdas = [i*0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    print(f"lambda {i+1}/{len(test_lamdas)}:{lamda}")
    A = lamda
    variances.append(get_IS_variance(lamda, 10000))
   # clear_output(wait=True)  change later

optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print(f"Optimal Lamda: {optimal_lamda}")
print(f"Optimal Variance: {IS_variance}")
print((IS_variance/10000)**0.5)


#ploting

plt.plot(test_lamdas[5:40], variances[5:40])
plt.title("Variance of MC at Diffentent Lamda Values")
plt.show()


def importance_sampling_MC(lamda, num_samples):
    A = lamda
    running_total = 0
    for i in range(num_samples):
        r = get_rand_number(0,1)
        running_total += f_of_x(inverse_G_of_r(r, lamda=lamda))/g_of_x(inverse_G_of_r(r, lamda=lamda), A, lamda)

    approximation = float(running_total/num_samples)

    return approximation


#run simulation

num_samples = 10000
approx = importance_sampling_MC(optimal_lamda, num_samples)
variance= get_IS_variance(optimal_lamda, num_samples)
error = (variance/num_samples)**0.5




#display results


print(f"Importance Sampling Approxiamtion: {approx}")
print(f"Variance: {variance}")
print(f"Error: {error}")

