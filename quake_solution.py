#! /usr/bin/env -S python3 -i
# quake_solution.py ===========================================================

import numpy as np
import quake
from multivarious.utl import StableNamespace

# ----- design variables --------------------------------------------------
#                      K   Fult   H
v_init = np.array([  145.0 , 28.0 , 4.72 ])   # initial guess (final optimized design)
v_lb   = np.array([  50 , 5 , 3 ])   #  lower bounds
v_ub   = np.array([  2000 , 100 , 10 ])   #  upper bounds


# ---- constants ----------------------------------------------------------
cts = quake.default_constants()

# ---- random seed strategy -----------------------------------------------
# if cts.seed == None, realistically, a new seed for every earthquake ground motion
# otherwise ,    unrealistically, the same seed for reproducible results
#                useful for a quick optimization, but not realistic

# ---- plot and evaluate the initial design -------------------------------
cts.plots = 0  # animate to evaualte the initial design -- animatiosn make this take longer so I set it to 0, set to 2 for animations on
f_init , g_init = quake.analysis( v_init , cts )
   
print(f'\ninitial cost        = {f_init:.6g} cu.m')
print(f'initial constraints = {g_init}')
   
if input('\n  OK to continue to gridded search? [y]/n : ').strip().lower() == 'n':
   print('    . . . sure, no problem, skipping the gridded search.\n')
else:
    # values of the design variables in the gridded search
    search_set = StableNamespace(
        K_set = np.linspace( 50 , 1000 , 2 ) ,
        F_set = np.linspace( 5 , 50 , 2 ) ,
        H_set = np.linspace( 3 , 4 , 2 )
    )

    cts.plots = 0  # animate to evaualte the initial design
    cts.fig_num = 1050  # animate to evaualte the initial design
    quake.search( search_set , cts , re_analysis = True )

if input('  OK to continue to design optimization? [y]/n : ').strip().lower() == 'n':
    print('    . . . sure, no problem, skipping the optimization.\n')
    v_opt = v_init
else:
    # ---- run the design optimization ------------------------------------
    # ---- optimization options
    #  index:   0    1      2      3      4          5      6     7      8
    #  name:   msg  tol_v  tol_f  tol_g  max_evals  pnlty  expn  m_max  cov_F
    options = [ 2,  0.10,  0.1,  0.01,   2000,      1.0,   2.0,  50,    0.20 ]

    cts.phi_D = 2.25  # demand safety factor > 1.0
    cts.plots = 0    # no plots or animations during the optimization
    cts.fig_num = 1100  # animate to evaualte the initial design
    v_opt, f_opt, g_opt = quake.optimize( v_init , v_lb, v_ub, options, cts )

    # ---- show and confirm the nominal optimal design --------------------
    cts.plots = 0  # animate to evaluate the optimized result
    cts.fig_num = 1200  # animate to evaualte the initial design
    f_opt, g_opt = quake.analysis( v_opt , cts )
   
if input('  OK to continue to uncertainty analysis? [y]/n : ').strip().lower() == 'n':
    print('    . . . sure, no problem, skipping the uncertainty analysis.\n')
else:
    # ---- run the uncertainty analysis -----------------------------------
    n_sims      = 1000  # total number of Monte Carlo simulations -- This takes a while, maybe 5 minutes? (d.m)
    re_analysis = True  # True: run simulations; False: load saved data
    cts.phi_D   = 1.0   # demand safety factor = 1.0
    cts.plots   = 0     # no plots or animation during the Monte Carlo loop
    cts.fig_num = 1300  # animate to evaualte the initial design

    P_topple , P_collapse  = quake.uncertainty( v_opt, cts, n_sims )

    print(f'\nProbability of collapse = {P_collapse:.4f}  ({P_collapse*100:.2f}%)')
    print(f'Probability of toppling = {P_topple:.4f}  ({P_topple*100:.2f}%)')

print('. . . all done.')