#! /usr/bin/env -S python3 -i
# quake_solution.py ===========================================================

import numpy as np
import quake 
from multivarious.utl import StableNamespace

# ----- design variables --------------------------------------------------
#                      K   Fult   H
v_init = np.array([  ... , ... , ... ])   # initial guess
v_lb   = np.array([  ... , ... , ... ])   #  lower bounds
v_ub   = np.array([  ... , ... , ... ])   #  upper bounds

# ---- constants ----------------------------------------------------------
cts = quake.default_constants()

# ---- random seed strategy -----------------------------------------------
# if cts.seed == None, realistically, a new seed for every earthquake ground motion 
# otherwise ,    unrealistically, the same seed for reproducible results
#                useful for a quick optimization, but not realistic

# ---- plot and evaluate the initial design ------------------------------- 
cts.anim = 2  # draw animation to evaualte the initial design
f_init, ... = quake.analysis( ... , ... )
    
print(f'\ninitial cost        = {f_init:.6g} cu.m')
print(f'initial constraints = {g_init}')
    
if input('\n  OK to continue to gridded search? [y]/n : ').strip().lower() == 'n':
   print('    ... sure, no problem, skipping the gridded search.\n')
else:
    # values of the design variables in the gridded search
    search_set = StableNamespace(
        K_set = np.linspace( ... , ... , ... ) , 
        F_set = np.linspace( ... , ... , ... ) , 
        H_set = np.linspace( ... , ... , ... )
    )

    quake.search( search_set , cts , re_analysis = False )

if input('  OK to continue to design optimization? [y]/n : ').strip().lower() == 'n':
    print('    ... sure, no problem, skipping the optimization.\n')
    v_opt = v_init
else: 
    # ---- run the design optimization ------------------------------------
    # ---- optimization options 
    #  index:   0    1      2      3      4          5      6     7      8
    #  name:   msg  tol_v  tol_f  tol_g  max_evals  pnlty  expn  m_max  cov_F
    options = [ 2,  0.01,  0.05,  0.01,  500,       1.0,   2.0,  50,    0.20 ]

    cts.phi_D = 1.5  # demand safety factor > 1.0
    cts.anim = 0     # do not draw animations during the optimization 
    ..., ..., ... = quake.optimize( ... , ... ... etcc )

    # ---- show and confirm the nominal optimal design -------------------- 
    cts.anim = 2  # animate to evaluate the optimized result
    ..., ... = quake.analysis( ... , ... )
    
if input('  OK to continue to uncertainty analysis? [y]/n : ').strip().lower() == 'n':
    print('    ... sure, no problem, skipping the uncertainty analysis.\n')
else: 
    # ---- run the uncertainty analysis -----------------------------------
    n_sims      = 1000  # total number of Monte Carlo simulations
    re_analysis = True  # True: run simulations; False: load saved data
    cts.phi_D   = 1.0   # demand safety factor = 1.0
    cts.anim    = 0     # no plots or animation during the Monte Carlo loop

    ... ,  ...  = quake.uncertainty ... , ... , ... , ... , fig_no= ...  )

    print(f'\nProbability of collapse = {P_collapse:.4f}  ({P_collapse*100:.2f}%)')
    print(f'Probability of toppling = {P_topple:.4f}  ({P_topple*100:.2f}%)')

print('... all done.')