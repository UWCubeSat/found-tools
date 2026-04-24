"""
This one might actually work?

PSEUDOCODE
"""

"""
inputs: 
- pos_vec, 3-vector of position estimate from FOUND
- horizon_pts, list of length n of 2-vectors of horizoon points
    - these are in image coordinates
    - can be made homogenous by appending a 1

- A_c from FOUND
- 

    
process:
- compute H_i = 2 * pos_vec^T ()
- compute variance of epsilon_i
- compute then invert P_sc = sum_i H_i^T sigma_i^-2 H_i


output: covariance matrix of pos_vec
"""