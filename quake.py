#! /usr/bin/env -S python3 -i
"""
quake.py

A Python moddule to analyze, animate, optimixe, and assess 
the response of an inelastic building and its contents to seismic shaking. 

Typical usage
-------------
    from multivarious.utl import StableNamespace
    import numpy as np
    import quake

    cost, constraints = quake.analysis( v, cts )

    # design variables (v) 
    v = np.array([500.0, 25.0, 3.0])   # K (kN/m), Fult (kN), H (m)

    # analysis constants (cts) 
    cts = StableNamespace(
          g=9.81, M=1.0, E=2e8, Sy=250e3, k=0.05,
          wBlk=0.4, hBlk=1.2, PGA=3.5,
          fg=1.5, zg=0.9, aa=4.0, tau=2.0,
          phi_C = 1.0, phi_D = 1.0, 
          t=np.arange(3000)*0.01,
          anim=1,
    )

H.P. Gavin, Dept. Civil and Environ. Eng'g, Duke Univ.
matlab: 2018-05-19, 2020-04-09, 2022-04-21  python: 2026-04-07
"""

#< import packages ------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

from multivarious.dsp import eqgm_1d
from multivarious.ode import ode4u
from multivarious.opt import ors, nms, sqp
from multivarious.rvs import lognormal
from multivarious.utl import StableNamespace, format_plot, plot_cvg_hst, plot_scatter_hist
#>


#< Log-space helpers for the high-order rocking-block penalty terms.
# Direct computation of d**(n-1) with n=42, or (d/(Ï€/2))**10, overflows
# float64 when the ODE integrator lets d run away on an unstable trajectory.
# These helpers exponentiate via log-space and clip the exponent to Â±500,
# giving a maximum magnitude of e^500 â‰ˆ 5e217 â€” physically unreachable but
# finite, so the integrator can recover rather than propagate inf or nan.

def _pow_odd( x, p, clip=500.0 ):
    """Sign-preserving safe power x**p for odd integer p, via log-space."""
    if x == 0.0:
        return 0.0
    return np.sign(x) * np.exp(np.clip(p * np.log(abs(x)), -clip, clip))

def _pow_even( x, p, clip=500.0 ):
    """Safe non-negative power x**p for even integer p, via log-space."""
    if x == 0.0:
        return 0.0
    return np.exp(np.clip(p * np.log(abs(x)), -clip, clip))
#>

def default_constants():
    cts = StableNamespace(
        g     =   9.81,    # gravitational acceleration, m/s^2
        M     =  10.0,     # mass of structure, ton
        E     = 200.0e6,   # Young's modulus of steel, kN/sq.m
        Sy    = 380.0e3,   # yield stress of steel, kN/sq.m
        k     =   0.02,    # post-yield stiffness ratio
        wBlk  =   0.25,    # width of rocking block, m
        hBlk  =   1.0,     # height of rocking block, m
        PGA   =   3.5,     # peak ground acceleration, m/sq.m
        fg    =   1.5,     # ground motion frequency, Hz
        zg    =   0.9,     # ground motion damping ratio
        aa    =   4.0,     # earthquake envelope rise-time exponent  (eqgm_1d: aa) 
        tau   =   2.0,     # earthquake envelope decay time, s       (eqgm_1d: ta) 
        phi_C =   1.0,     # capacity safety factor
        phi_D =   1.0,     #  demand  safety factor
        Dep   =   0.1,     # elastic-plastic displacement, m  
        c     =   0.1,     # viscous damping coefficient
        l     =   None,    # block corner to center of gravity length, see below, m
        alpha =   None,    # block balance angle, see below, rad
        t     =   None,    # time vector for simulations, see below, sec
        v     =   None,    # array of design variables, updated later
        randomize_demands = True, # randomize PGA and ta in each earthquake
        seed  =   None,    # random number generator seed
        anim  =   2,       # 1: draw animation, 0: no animation
    )
    # ---- update derived constants within the StableNamespace cts
    cts.l     = 0.5 * np.sqrt(cts.wBlk**2 + cts.hBlk**2)   # corner-to-CG length, m
    cts.alpha = np.arctan(cts.wBlk / cts.hBlk)             # balance angle, rad

    # time vector for earthquake ground motion 
    T    = 30.0                   # total time span, s
    dt   =  0.010                 # time step, s
    N    = int(np.floor(T / dt))  # number of time values
    cts.t = np.arange(N) * dt     # include t into the analysis constants 

    return cts

# ---------------------------------------------------------------------------
def system( t_i, x, u, cts ):
    """ Differential equations of motion for inelastic SDOF + rocking block.

    Called by ode4u at each time step.  Implements a Bouc-Wen hysteresis
    model for the primary structure and a nonlinear rocking model for the
    secondary block.

    INPUTS
    ------
    t_i : float
        Current time, s  (not used explicitly; ground input comes via u)
    x   : ndarray, shape (5,)
          State vector:
            x[0]  D   structural displacement, m
            x[1]  V   structural velocity, m/s
            x[2]  d   block rotation, rad
            x[3]  v   block angular velocity, rad/s
            x[4]  z   Bouc-Wen hysteresis variable
    u   : ndarray, shape (1,)
          Ground acceleration at current time step, m/s^2  (sampled by ode4u)
    cts : StableNamespace
          System constants (see the analysis function for full field list)

    OUTPUTS
    -------
    dxdt : ndarray, shape (5,)
           State derivative
    y    : ndarray, shape (1,)
           Restoring force R(t), kN
    """
    # dynamic states --------------------------------------------------------
    D = x[0]        # structural displacement, m
    V = x[1]        # structural velocity, m/s
    d = x[2]        # block rotation, rad
    v = x[3]        # block angular velocity, rad/s
    z = x[4]        # Bouc-Wen hysteresis variable

    ag = u[0]     # scalar ground acceleration at this step, m/s^2

    c_blk = 1e7 / np.sqrt(cts.l * cts.g)  # block viscous damping, kN s/rad
    n     = 42                        # even exponent for toppled-block penalty

    # extract the design variables from the constants 
    K    = cts.v[0]
    Fult = cts.v[1]
    H    = cts.v[2]

    # hysteretic restoring force, kN
    R = cts.k * K * D + (1.0 - cts.k) * Fult * z + cts.c * V

    # instability guards: structure collapsed or block irreversibly toppled
    if abs(D) > 1.5*H: # or abs(d) > np.pi:
        return np.zeros(5), np.array([R])

    # translational acceleration of structure, m/s^2 ... eqn (11)
    A = cts.g * D / H - R / cts.M - ag 

    # Bouc-Wen hysteresis rate equation ... eqn (12)
    dzdt = (1.0 - z**2 * (0.5*np.sign(V*z) + 0.5)) * V / cts.Dep

    # energy dissipation at block impact (rotation passing through zero)
    if abs(d) < 1e-3:
        v = 0.20 * v

    # do not initiate rocking until floor acceleration exceeds tipping threshold
    if abs(A + ag) < cts.g * np.tan(cts.alpha) and abs(v) < 1e-6:
        v = 0.0
        a = 0.0
    else:
        # rotational acceleration of block, rad/s^2  ... eqn (18)
        a = (
             (1.0 / cts.l) * (A + ag) * np.cos(cts.alpha - abs(d))
             - (cts.g / cts.l) * np.sin(cts.alpha - abs(d)) * np.sign(d)
             - (n - 1) * (2.0/np.pi)**n * _pow_odd(d, n - 1)
             - c_blk * _pow_even(d / (np.pi/0.5), 10) * v
        )

    dxdt = np.array([V, A, v, a, dzdt])

    return dxdt, R  


# ----------------------------------------------------------------------------
def analysis( v, cts = default_constants() ):
    """ Simulate the response of an inelastic SDOF system to an earthquake.

    The primary SDOF system (inelastic beam-column) supports a rocking block
    that can topple.  A synthetic earthquake ground motion is generated via
    eqgm_1d and the coupled equations of motion are integrated with ode4u.

    INPUTS
    ------
    v : array-like, length 3
        design variables
        v[0] = K      column stiffness , kN/m
        v[1] = Fult   column ultimate strength , kN
        v[2] = H      column height , m

    cts : StableNamespace
          Analysis constants - see field list below.
          cts.g      gravitational acceleration, m/s^2
          cts.M      mass of structure, ton
          cts.E      Young's modulus of steel, kN/sq.m
          cts.Sy     yield stress of steel, kN/sq.m
          cts.k      post-yield stiffness ratio
          cts.wBlk   width of the rocking block, m
          cts.hBlk   height of the rocking block, m
          cts.PGA    peak ground acceleration, m/s^2
          cts.fg     ground motion frequency, Hz
          cts.zg     ground motion damping ratio
          cts.aa     earthquake envelope rise-time parameter   (eqgm_1d: aa)
          cts.tau    earthquake envelope decay time constant, s (eqgm_1d: tau)
          cts.t      time vector for the ground motion, s
          cts.anim   2: draw animation, 1: plots, 0 no animation

    OUTPUTS
    -------
    cost : float
        Volume of steel in the columns, cu.m
    constraints : ndarray, shape (2,)
        [0]  max structure displacement : min(4 Dep, 1.0)  [collapse margin]
        [1]  max block rotation : pi/3                     [toppling margin]
    """

    # ---- design variables
    K    = v[0]          # stiffness, kN/m
    Fult = v[1]          # ultimate strength, kN
    H    = v[2]          # column height, m

    # ---- inelastic beam-column section dimensions from design variables
    Dep   = Fult / K                        # elastic-to-plastic deflection, m
    depth = 0.5 * H**2 * cts.Sy / cts.E * K / Fult  # cross section depth 
    width = K / cts.E * (H / depth)**3              # cross section width 
    c     = 0.02 * 2.0 * np.sqrt(cts.M * K)  # viscous damping rate, kN/(m/s)

    # ---- update derived constants computed from design variables in cts
    cts.Dep   = Dep
    cts.c     = c

    # ---- include the current values of the design variables v in constants cts
    cts.v = v

    # ---- lognormal random samples for PGA and tau ----------------------- 
    PGA, tau = cts.PGA, cts.tau
    if cts.randomize_demands: 
        PGA = lognormal.rnd(cts.phi_D * PGA, 0.30, 1, 1)
        tau = lognormal.rnd(            tau, 0.30, 1, 1)

    # ---- use eqgm_1d to generate synthetic earthquake ground motion 
    ground_accel, ground_veloc, ground_displ, _, _, _, _  = eqgm_1d(
        PGA, cts.fg, cts.zg, cts.aa, tau, cts.t, fig_no=0, seed=cts.seed)

    PGV = np.max(np.abs(ground_veloc))

    # ---- use ode4u to integrate equations of motion
    #      ode4u has shape (m, p); accel has shape (1, n) from eqgm_1d
    #      returns: time (1,p), x (n,p), x_dot (n,p), y (m,p)
    x0 = np.zeros(5)
    t, x, x_dot, y = ode4u(system, cts.t, x0, ground_accel, cts )

    R = y[0,:]                          # restoring force, kN,  shape (p,)

    # ---- extract peak responses 
    maxD = np.max(np.abs(x[0, :]))      # peak structure displacement, m
    maxd = np.max(np.abs(x[2, :]))      # peak block rotation, rad

    # ---- compute design objective to be minimized 
    objective = depth * width * H       # steel volume, m

    # ---- compute design constraints
    constraints = np.array([
        maxD - min(4.0*Dep, 1.0),       # collapse margin (> 0: violated)
        maxd - cts.alpha/2,             # toppling margin (> 0: violated)
              -PGV                      # extract negative PGV via constraints
    ])

    # ---- display animation
    if cts.anim > 0:
        speed = 2
        animate( t, ground_displ, ground_accel, x, x_dot, R, v, cts, speed )

    return objective, constraints


# ----------------------------------------------------------------------------
def optimize( v_init, v_lb, v_ub, options, cts ):
    """ Optimize an inelastic SDOF building and contents subjected to an
    earthquake.
    Minimize steel volume subject to collapse and toppling constraints.

    Solves:   min  f(K, Fult, H)       ... design objective 
              s.t. g(K, Fult, H) <= 0  ... design constraints 

    using the Nelder-Mead simplex optimizer (nms) from multivarious.opt.

    Parameters:
    ---- design variabls ------------------------------------------------------
    v_init   initial guess for [ stiffness, strength, height ]
    v_lb     lower bound for [ stiffness, strength, height ] 
    v_ub     upper bound for [ stiffness, strength, height ] 

    ---- analysis constants ---------------------------------------------------
    cts   StableNamespace of constants ... see the analysis function for details
    """

    # ---- run the optimization 
#   v_opt, f_opt, g_opt, cvg_hst, _, _ =  ... ... ... 

    # ---- plot convergence history 
    plot_cvg_hst(cvg_hst, v_opt, options, save_plots=True)

    return v_opt, f_opt, g_opt


# ----------------------------------------------------------------------------
def uncertainty( v_opt, cts, n_sims=1000, re_analysis=True, fig_no=30 ):
    """ Monte Carlo uncertainty analysis of the inelastic SDOF earthquake response.

    Examines the safety of the system described by earthquake_analysis.py
    in terms of excessive displacement of the primary mass (collapse) and
    toppling of the secondary rocking block.

    PGA and earthquake rise_time (aa*ta) are treated as independent lognormal
    random variables with coefficient of variation 0.30.

    Usage
    -----
    Set  re_analysis = True   to run (or re-run) all simulations and save.
    Set  re_analysis = False  to load a previous run and just re-plot.

    """

    #   timestamp = datetime.now ().strftime(â€™%Y%m%dT%H%M%Sâ€™)
    SAVE_FILE = 'quake-uncertainty.npz'

    # ========================================================================= 
    if re_analysis:

        # ---- lognormal random samples for PGA and tau ----------------------- 
        r = lognormal.rnd(
            mednX = np.array([cts.PGA, cts.tau]),
            covnX = np.array([0.30,   0.30]),
            N     = n_sims,
            R     =  np.eye(2),  # no correlation between PGA and ta
        )
        PGA_rand = r[0, :]       # sample of random PGA values (n_sims,)
        tau_rand = r[1, :]       # sample of random tau values (n_sims,)

        # ---- Monte Carlo loop ----------------------------------------------- 
        f = np.zeros(n_sims)            # cost at each simulation
        g = np.zeros((3, n_sims))       # constraints at each simulation and PGV

        t_start = time.perf_counter()

        for sim in range(n_sims):

            cts.PGA = PGA_rand[sim]      # randomised peak ground acceleration
            cts.tau = tau_rand[sim]      # randomised envelope decay time

            f[sim], g[:, sim] = analysis( v_opt, cts )

            elapsed   = time.perf_counter() - t_start
            secs_sim  = elapsed / (sim + 1)
            secs_left = int(round((n_sims - sim - 1) * secs_sim))
            eta       = datetime.now() + timedelta(seconds=secs_left)

            if (sim + 1) % 10 == 0:
                print(f'  {sim+1:4d} ({100*(sim+1)/n_sims:5.1f}%)'
                      f'  {secs_sim:5.2f} s/sim'
                      f'  eta: {eta.strftime("%H:%M:%S")} ({secs_left:4d} s)'
                      f'  P_c: {(np.sum(g[0,0:sim]>0)/sim):5.3f}'
                      f'  P_t: {(np.sum(g[1,0:sim]>0)/sim):5.3f}'
                )

        # ---- save results --------------------------------------------------- 
        np.savez(SAVE_FILE, f=f, g=g, r=r)
        print(f'\nUncertainty analysis results saved to {SAVE_FILE}')

    # ========================================================================= 
    # Load results (whether just computed or from a previous run)

    data     = np.load(SAVE_FILE)
    f        = data['f']                    # design objective   (n_sims,)
    g        = data['g']                    # design constraints (2, n_sims)
    r        = data['r']                    # random values      (2, n_sims)
    PGA_rand = r[0, :]
    tau_rand = r[1, :]
    PGV_rnd  = -g[2, :]                     # peak ground velocity

    # ---- failure probabilities computed from constraint values 
    P_collapse = np.sum(g[0, :] > 0) / n_sims
    P_topple   = np.sum(g[1, :] > 0) / n_sims

    # ---- effective earthquake rise_time computed from system constant tau
    rise_time = cts.aa * tau_rand     # rise time s
    #   duration = 5.7 * cts.aa**0.42 * tau_rand     # s
    # ========================================================================= 
    # Plot histograms and scatter plots of constraints and random values   
    n_bins = max(10, int(np.floor(n_sims / 20)))

    r_idx = [ 0, 1 ]  # random values to plot
    g_idx = [ 0, 1 ]  # constraints to plot , collapse and topple
    InData = np.block([ [PGA_rand] , [PGV_rnd] , [rise_time] ])

    # names for "input, X" values (r) and "output Y" values (g) 
    xy_names = { 'X': [ 'PGA', 'PGV', 'rise time' ] ,
                 'Y': [ 'collapse', 'topple'] }

    plot_scatter_hist( InData, g[g_idx,:], fig_no,
                       var_names=xy_names, font_size=15, ci=0.90)

    return P_topple, P_collapse


# ----------------------------------------------------------------------------
def search( variable_set, cts, re_analysis = True ):
    """ Gridded search across all combinations of design variables values
    provided in the StableNamespace variable_set, and plotting the results 
    as a set of 2D plots for variables K_set and F_set, one plot for each 
    value of H_set.  
    cts is a StableNamespace containing all constants for this module. 

    returns None
    """

    # Parameter sets
    K_set = variable_set.K_set
    F_set = variable_set.F_set
    H_set = variable_set.H_set

    # Number of values in each set
    nK = len(K_set)
    nF = len(F_set)
    nH = len(H_set)
    nQ = 100

    NumberOfSimulations = nK * nF * nH * nQ
    print(f"     Number of gridded search simulations: {NumberOfSimulations}\n")

    #   timestamp = datetime.now ().strftime(â€™%Y%m%dT%H%M%Sâ€™)
    SAVE_FILE = 'quake-search.npz'

    if re_analysis:
        # run nQ analyses for each and every combination of K, F, H
        # use the same set of nQ PGA and tau values for each combination of K, F, H
        # but each
        Pf1_set = np.zeros((nK, nF, nH))
        Pf2_set = np.zeros((nK, nF, nH))

        PGA_rand = lognormal.rnd(cts.PGA, 0.3, nQ)  # nQ PGA values
        tau_rand = lognormal.rnd(cts.tau, 0.3, nQ)  # nQ tau values

        cts.anim = 0  # no plots

        counter = 0
        start_time = time.time()
        for i in range(nK):
            for j in range(nF):
                for k in range(nH):
                    g1_sum = 0
                    g2_sum = 0
                    for l in range(nQ):
                        # run nQ analyses to estimate Pf1=P[g1 > 0] and Pf2=P[g2 > 0]

                        cts.PGA = PGA_rand[l]
                        cts.tau = tau_rand[l]

                        f, g = analysis([K_set[i], F_set[j], H_set[k]], cts)

                        if g[0] > 0:
                            g1_sum += 1
                        if g[1] > 0:
                            g2_sum += 1

                    Pf1_set[i, j, k] = g1_sum / (nQ+1)
                    Pf2_set[i, j, k] = g2_sum / (nQ+1)

                    counter += nQ
                    secs = time.time() - start_time
                    secs_per_sim = secs / counter
                    secs_left = round((NumberOfSimulations - counter) * secs_per_sim)
                    eta = datetime.now() + timedelta(seconds=secs_left)
                    eta_str = eta.strftime('%H:%M:%S')

                    print(f"{counter:6d} ({100*counter/NumberOfSimulations:5.1f}%) "
                          f"{secs_per_sim:5.2f} sec/sim  eta: {eta_str} "
                          f"Pfs= {Pf1_set[i, j, k]:5.3f} Pfb= {Pf2_set[i, j, k]:5.3f} "
                          f"{i:2.0f} {j:2.0f} {k:2.0f}" )

        # Save results to .npz file
        np.savez(SAVE_FILE, Pf1_set=Pf1_set, Pf2_set=Pf2_set, K_set=K_set, F_set=F_set, H_set=H_set )
        print(f'\nGridded search results saved to {SAVE_FILE}')

    else:
        # Load data saved from a previous gridded-search analysis
        data     = np.load(SAVE_FILE)
        Pf1_set  = data['Pf1_set']                      # Prob [ g1 > 0 ]
        Pf2_set  = data['Pf2_set']                      # Prob [ g2 > 0 ]
        K_set    = data['K_set'].flatten()              # column stiffness 
        F_set    = data['F_set'].flatten()              # column strength
        H_set    = data['H_set'].flatten()              # column height

    T_set = (2*np.pi)* np.sqrt(cts.M / K_set) 
    V_set = F_set / ( cts.M * cts.g )

    # Plotting
    format_plot(font_size=18, line_width=4, marker_size=16)
    plt.ion()
    for hv in [0, nH-1]:
        # Find indices matching conditions
        K1_05, F1_05 = np.where(Pf1_set[:, :, hv] < 0.05)
        K1_50, F1_50 = np.where(Pf1_set[:, :, hv] > 0.50)
        K2_30, F2_30 = np.where(Pf2_set[:, :, hv] < 0.30)
        K2_85, F2_85 = np.where(Pf2_set[:, :, hv] > 0.85)

        plt.figure(hv + 501, figsize = (8, 6))
        plt.clf()
        plt.plot(1050-K_set[K1_50], V_set[F1_50], 'or', label='Collapse > 50%',mew=4,mfc='w')
        plt.plot(1050-K_set[K1_05], V_set[F1_05], 'og', label='Collapse < 5%',mew=4,mfc='w')
        plt.plot(1050-K_set[K2_85], V_set[F2_85], '+r', label='Topple > 85%',mew=4,mfc='w')
        plt.plot(1050-K_set[K2_30], V_set[F2_30], '+g', label='Topple < 30%',mew=4,mfc='w')
        x_ticks = np.linspace(1000., (2*np.pi/2.0)**2*cts.M, 6)
        p_ticks = 2*np.pi * np.sqrt(cts.M / (1050-x_ticks) )
        x_labels = [f'{p:1.2f}' for p in p_ticks]
        plt.xticks( x_ticks, x_labels )
        plt.title(rf'$H$ = {H_set[hv]:.3f} m')
        plt.xlabel(r'natural period, $T_n = 2 \pi (M/K)^{1/2}$, s')
        plt.ylabel(r'strength-to-weight, $F_{ULT}$ / $(Mg)$')
        plt.savefig(f'quake-search-{hv+1}-A.pdf', bbox_inches='tight')

        plt.figure(hv + 502, figsize = (8, 6))
        plt.clf()
        plt.plot(K_set[K1_50], F_set[F1_50], 'or', label='Collapse > 50%',mew=4,mfc='w')
        plt.plot(K_set[K1_05], F_set[F1_05], 'og', label='Collapse < 5%',mew=4,mfc='w')
        plt.plot(K_set[K2_85], F_set[F2_85], '+r', label='Topple > 85%',mew=4,mfc='w')
        plt.plot(K_set[K2_30], F_set[F2_30], '+g', label='Topple < 30%',mew=4,mfc='w')
        plt.xlabel(r'column stiffness, $K$, kN/m')
        plt.ylabel(r'strength, $F_{ULT}$, kN')
        plt.savefig(f'quake-search-{hv+1}-B.pdf', bbox_inches='tight')

    # plt.show()

    return None


# ----------------------------------------------------------------------------
def animate( t, dg, ag, x, x_dot, R, v, cts, speed=2 ):
    """ Animate the response of the inelastic SDOF + rocking block system.

    INPUTS
    ------
    t     : ndarray (p,)       time vector, s
    dg    : ndarray (p,)       ground displacement, m
    ag    : ndarray (p,)       ground acceleration, m/sÂ²
    x     : ndarray (5, p)     state matrix  [D; V; d; v; z]
    x_dot : ndarray (5,p)      state derivative matrix
    R     : ndarray (p,)       hysteretic restoring force, kN
    v     : array-like, (3,)   [K, Fult, H]
    cts   : StableNamespace    analysis constants, see analysis()
    speed : int                animation speed
    """

    K    = v[0]  # stiffness
    Fult = v[1]  # strength
    H    = v[2]  # height

    dg = dg.flatten()
    ag = ag.flatten()
    R  =  R.flatten()

    w    = 5.0*cts.wBlk                        # box half-width  for animation, m
    doff = cts.wBlk/2                          # box half-height for animation, m
    z_wall = np.linspace(doff, H - doff, 20)   # vertical axis for spring profile

    # ---- re-compute section dimensions for annotation
    # ---- inelastic beam-column section dimensions from design variables
    Dep   = Fult / K  # elastic-plastic displacment 
    depth = 0.5 * H**2 * cts.Sy / cts.E * K / Fult    # cross section depth, m
    width = K / cts.E * (H / depth)**3                # cross section width, m

    Tn    = 2.0 * np.pi * np.sqrt(cts.M / K)          # natural period, s
    # tipping acceleration, m/^2
    A_tip    = cts.g * np.tan(cts.alpha) 
    # toppling acceleration, m/s^2
    A_top    = cts.g * cts.alpha * np.sqrt(1.0 + cts.l/cts.g*(2*np.pi/Tn)**2)

    # ---- extract state time histories from (5, p) matrix
    D_ =  x[0, :]           # structural displacement, m
    V_ =  x[1, :]           # structural velocity, m/s
    d_ =  x[2, :]           # block rotation, rad
    v_ =  x[3, :]           # block angular velocity, rad/s
    z_ =  x[4, :]           # hysteresis variable
    A_ =  x_dot[1, :] + ag  # total structural acceleration, m/s^2

    minD = min(np.min(dg) - w, np.min(dg + D_) - w)
    maxD = max(np.max(dg) + w, np.max(dg + D_) + w)

    fs = 14;  ms = 9;  lw = 7
    cg = [0.0, 0.6, 0.0]    # green  (ground)
    co = [1.0, 0.6, 0.0]    # orange (block / tipping threshold)

    # block outline in local (un-rotated) coordinates
    xBlk = cts.wBlk * np.array([-1, -1, 1,  1, -1]) / 2.0
    yBlk = cts.hBlk * np.array([ 0,  1, 1,  0,  0])

    sc = 0.15 * cts.alpha   # text offset for labels of +/-alpha and +/-Dep 

    # ---- helper: box x-coords (half-width w centred at cx)
    def box_x(cx):
        return cx + np.array([+w, -w, -w, +w, +w])

    # ---- helper: rotated rocking block outline
    def block_box(it):
        d_i  = d_[it]
        Rd   = np.array([[ np.cos(d_i), -np.sin(d_i)],
                         [ np.sin(d_i),  np.cos(d_i)]])
        cx   = dg[it] + D_[it]
        cy   = H + doff
        # pivot the block slightly off-centre depending on lean direction
        if d_i < 0:
            local = np.vstack([xBlk - 0.125, yBlk])
            return np.array([[cx + 0.125], [cy]]) + Rd @ local
        else:
            local = np.vstack([xBlk + 0.125, yBlk])
            return np.array([[cx - 0.125], [cy]]) + Rd @ local

    # ---- helper: sinusoidal spring/damper wall profile
    def wall_x(it):
        return dg[it] + 0.5*D_[it] * (1.0 - np.cos(np.pi * z_wall / H))

    plt.ion()

    plt.rcParams['figure.dpi']     = 72
    plt.rcParams['font.size']      = fs
    plt.rcParams['axes.labelsize'] = fs
    plt.rcParams['axes.titlesize'] = fs

    # ================================================================
    # Figure 1 - displacements & rotations
    # ================================================================
    T     = cts.t[-1]                          # duration of simulation to plot
    fig1, ax1 = plt.subplots(num=1, clear=True, figsize=(10, 5))
    ax1.plot(t, dg,  '-g',         linewidth=3, label=rf'ground')
    ax1.plot(t, d_,  color=co,     linewidth=3, label=rf'block $\theta$, rad')
    ax1.plot(t, D_,  '-b',         linewidth=3, label=rf'structure $x$, m')
    ax1.plot([4, T], [ cts.alpha,  cts.alpha], '--', color=co)
    ax1.plot([4, T], [-cts.alpha, -cts.alpha], '--', color=co)
    ax1.plot([4, T], [ Dep,    Dep],  '--k')
    ax1.plot([4, T], [-Dep,   -Dep],  '--k')
    dgPh, = ax1.plot(t[0], dg[0], 'o', color=cg, markersize=ms, linewidth=lw)
    drPh, = ax1.plot(t[0], D_[0], 'o', color='b', markersize=ms, linewidth=lw)
    dbPh, = ax1.plot(t[0], d_[0], 'o', color=co,  markersize=ms, linewidth=lw)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-2.2*cts.alpha, 2.2*cts.alpha)
    ax1.set_ylabel(rf'relative displ. m , rotation, rad')
    ax1.legend()
    ax1.grid(True)
    ax1.text(22,  sc + cts.alpha, rf'$+\alpha$ = +{cts.alpha:5.3f} rad',
             color=co, fontweight='bold')
    ax1.text(22, -sc - cts.alpha, rf'$-\alpha$ = -{cts.alpha:5.3f} rad',
             color=co, fontweight='bold')

    # ================================================================
    # Figure 2 - accelerations
    # ================================================================
    fig2, ax2 = plt.subplots(num=2, clear=True, figsize=(10, 5))
    ax2.plot(t, ag, '-g',  linewidth=3, label=rf'ground $a_g$')
    ax2.plot(t, A_, '-b',  linewidth=3, label=rf'structure $A$')
    ax2.plot([4, T], [ A_tip,  A_tip], '--', color=co)
    ax2.plot([4, T], [-A_tip, -A_tip], '--', color=co)
    agPh, = ax2.plot(t[0], ag[0], 'o', color=cg, markersize=ms, linewidth=lw)
    arPh, = ax2.plot(t[0], A_[0], 'o', color='b', markersize=ms, linewidth=lw)
    ax2.set_xlim(0, T)
    ax2.set_ylim(min(np.min(A_), -1.5*A_tip), max(np.max(A_), 1.5*A_tip))
    ax2.set_ylabel(rf'total acceleration, m/s$^2$')
    ax2.legend()
    ax2.grid(True)
    ax2.text(22,  1.1*A_tip, rf'$+A_{{tip}}$ = +{A_tip:5.3f} m/s$^2$', color=co, fontweight='bold')
    ax2.text(22, -1.2*A_tip, rf'$-A_{{tip}}$ = -{A_tip:5.3f} m/s$^2$', color=co, fontweight='bold')

    if cts.anim < 2:
        plt.pause(0.1) # yield to event loop long enough to flush to screen
        return None

    # ================================================================
    # Figure 3 - hysteresis loop
    # ================================================================
    Rfixed = R - cts.M * cts.g * D_ / H      # net restoring force (gravity corrected)
    xlims  = [min(np.min(D_), -1.5*Dep), max(np.max(D_), 1.5*Dep)]
    fig3, ax3 = plt.subplots(num=3, clear=True, figsize=(7, 7))
    ax3.set_title(
        rf'$K$={K:6.1f} kN/m  $H$={H:4.2f} m  '
        rf'$d$={depth:4.2f} m  $b$={width:4.2f} m  '
        rf'$T_n$={Tn:4.1f} s  $A_t$={A_top:4.2f} m/sÂ²  PGA={cts.PGA:4.1f} m/sÂ²',
        fontsize=11
    )
    ax3.plot(D_, Rfixed, '-k')
    ax3.plot([xlims[0], 0],   [-Fult, -Fult], '--k')
    ax3.plot([0,  xlims[1]],  [ Fult,  Fult], '--k')
    ax3.plot([-Dep, -Dep], np.array([-Fult, +Fult])/10, '--b')
    ax3.plot([ Dep,  Dep], np.array([-Fult, +Fult])/10, '--b')
    ax3.plot([-Dep,  Dep], [-Fult,  Fult], '--b', linewidth=3)
    hPh, = ax3.plot(D_[0], Rfixed[0], 'o', color='c', markersize=ms, linewidth=lw)
    ax3.set_xlim(xlims)
    ax3.set_ylim(-1.1*Fult, 1.1*Fult)
    ax3.set_xlabel(rf'structural deflection $x(t)$, m')
    ax3.set_ylabel(rf'structural force $R(t)$, kN')
    ax3.text(-0.8*Dep, -0.9*Fult, rf'$Kx(t)$',             color='b', fontweight='bold')
    ax3.text(-Dep, 0.8*Fult, rf'$R(x(t))-Mgx(t)/H$',  fontweight='bold')
    ax3.text(-Dep,     Fult, rf'$+F_{{ult}}$ = +{Fult:5.2f} kN', fontweight='bold')
    ax3.text(+0.1*Dep, -Fult, rf'$-F_{{ult}}$ = +{Fult:5.2f} kN', fontweight='bold')

    # ================================================================
    # Figure 4 - animation
    # ================================================================
    ax_lim = max(0.6*max(-minD, maxD) + w, (1.1*H + 1.2) / 2.0)
    bB0    = block_box(0)
    w0     = wall_x(0)

    fig4, ax4 = plt.subplots(num=4, clear=True, figsize=(7, 7))
    wallLh, = ax4.plot(w0 - 0.8*w, z_wall, '-k', linewidth=lw)
    wallRh, = ax4.plot(w0 + 0.8*w, z_wall, '-k', linewidth=lw)
    boxGh,  = ax4.plot(box_x(dg[0]),       doff + np.array([+1,-1,-1,+1,+1])*doff,
                       color=cg, linewidth=lw+3)
    boxMh,  = ax4.plot(box_x(dg[0]+D_[0]), H    + np.array([+1,+1,-1,-1,+1])*doff,
                       '-b',   linewidth=lw+3)
    boxBh,  = ax4.plot(bB0[0, :], bB0[1, :], color=co, linewidth=lw+3)
    timeTxt =  ax4.text(w, H + 0.5, rf'$t$ = {t[0]:5.2f} s', fontsize=fs)
    ax4.set_xlim(-ax_lim,  ax_lim)
    ax4.set_ylim(-1.1*doff, H + cts.hBlk + 0.5)
    ax4.set_aspect('equal')

    # ================================================================
    # Blit setup
    # ================================================================
    # animated=True MUST be set before canvas.draw() so these artists are
    # excluded from the background buffer.  Without it the initial marker
    # positions are baked into the background, causing ghosting and extra work.
    animate_artists = [
        [dgPh, drPh, dbPh],                             # fig1 / ax1
        [agPh, arPh],                                   # fig2 / ax2
        [hPh],                                          # fig3 / ax3
        [wallLh, wallRh, boxGh, boxMh, boxBh, timeTxt], # fig4 / ax4
    ]
    for group in animate_artists:
        for artist in group:
            artist.set_animated(True)

    figs = [fig1, fig2, fig3, fig4]
    axes = [ax1,  ax2,  ax3,  ax4]
    for fig in figs:
        fig.canvas.draw()                      # static content only
    plt.pause(0.05)                            # let windows appear
    bgs = [fig.canvas.copy_from_bbox(ax.bbox)  # per-axis bbox: fewer pixels
           for fig, ax in zip(figs, axes)]

    # ================================================================
    # Animation loop
    # ================================================================
    Nt      = len(cts.t)                    # points of simulation to animate
    box_y_G = np.array([+doff, +doff, -doff, -doff, +doff])
    box_y_M = H + np.array([+doff, +doff, -doff, -doff, +doff])

    for it in range(0, Nt-1, speed):

        cw = 'r' if abs(z_[it]) > 0.70 else 'k'   # red = yielding

        w_prof = wall_x(it)
        bB     = block_box(it)

        # Fig 1
        dgPh.set_xdata([t[it]]);  dgPh.set_ydata([dg[it]])
        drPh.set_xdata([t[it]]);  drPh.set_ydata([D_[it]])
        dbPh.set_xdata([t[it]]);  dbPh.set_ydata([d_[it]])
        # Fig 2
        agPh.set_xdata([t[it]]);  agPh.set_ydata([ag[it]])
        arPh.set_xdata([t[it]]);  arPh.set_ydata([A_[it]])
        # Fig 3
        hPh.set_xdata([D_[it]]);  hPh.set_ydata([Rfixed[it]])
        hPh.set_color(cw)
        # Fig 4
        wallLh.set_xdata(w_prof - 0.8*w);  wallLh.set_color(cw)
        wallRh.set_xdata(w_prof + 0.8*w);  wallRh.set_color(cw)
        boxGh.set_xdata(box_x(dg[it]));          boxGh.set_ydata(box_y_G)
        boxMh.set_xdata(box_x(dg[it] + D_[it])); boxMh.set_ydata(box_y_M)
        boxBh.set_xdata(bB[0, :]);  boxBh.set_ydata(bB[1, :])
        timeTxt.set_text(rf'$t$ = {t[it]:5.2f} s')

        for fig, ax, bg, artists in zip(figs, axes, bgs, animate_artists):
            fig.canvas.restore_region(bg)
            for artist in artists:
                ax.draw_artist(artist)
            fig.canvas.blit(ax.bbox)    # push only the axes pixels
        fig1.canvas.flush_events()      # one compositor flush per frame

    # ================================================================
    # Post-loop peak annotations
    # ================================================================
    # ---- restore all animated artists to normal rendering -----------------
    for group in animate_artists:
        for artist in group:
            artist.set_animated(False)
    for fig in [fig1, fig2, fig3, fig4]:
        fig.canvas.draw()               # full synchronous redraw, all artists
    plt.ioff()

    # Fig 1 â€” peak displacement and Â±Dep labels
    idx_D = np.argmax(np.abs(D_))
    ax1.plot(t[idx_D], D_[idx_D], '*k', markersize=9)
    ax1.text(t[min(idx_D+25, Nt-1)], D_[idx_D],
             rf'$|D|_{{peak}}$ = {np.abs(D_[idx_D]):5.3f} m',
             fontsize=14, color='b', fontweight='bold')
    ax1.text(22,  sc + Dep, rf'$+D_{{ep}}$ = +{Dep:5.3f} m', color='b', fontweight='bold')
    ax1.text(22, -sc - Dep, rf'$-D_{{ep}}$ = -{Dep:5.3f} m', color='b', fontweight='bold')

    # Fig 2 â€” peak structure acceleration and PGA
    idx_A = np.argmax(np.abs(A_))
    ax2.plot(t[idx_A], A_[idx_A], '*k', markersize=9)
    ax2.text(t[min(idx_A+25, Nt-1)], A_[idx_A],
             rf'$|A|_{{peak}}$ = {np.abs(A_[idx_A]):5.3f} m/sÂ²',
             fontsize=14, color='b', fontweight='bold')
    idx_g = np.argmax(np.abs(ag))
    ax2.plot(t[idx_g], ag[idx_g], '*k', markersize=9)
    ax2.text(t[min(idx_g+25, Nt-1)], ag[idx_g],
             rf'PGA = {np.abs(ag[idx_g]):5.3f} m/sÂ²',
             fontsize=14, color='g', fontweight='bold')

    # Fig 3 â€” Â±Dep labels on hysteresis plot
    ax3.text(-1.1*Dep,  0.2*Fult, rf'$-D_{{ep}}$ = -{Dep:5.3f} m', color='b', fontweight='bold')
    ax3.text(+0.2*Dep, -0.3*Fult, rf'$+D_{{ep}}$ = +{Dep:5.3f} m', color='b', fontweight='bold')

    for fig in [fig1, fig2, fig3, fig4]:
        fig.canvas.draw_idle()

    return None


############################################################################
if __name__ == '__main__':
    # run a sample analysis 
    import numpy as np

    # example of running the analysis function ... 
    cts = default_constants()
    # override demo values (keep full StableNamespace from default_constants)
    cts.g = 9.81
    cts.M = 1.0
    cts.E = 2.0e8
    cts.Sy = 250.0e3
    cts.k = 0.05
    cts.wBlk = 0.4
    cts.hBlk = 1.2
    cts.PGA = 3.5
    cts.fg = 1.5
    cts.zg = 0.9
    cts.aa = 4.0
    cts.tau = 2.0
    cts.phi_C = 1.0
    cts.phi_D = 1.0
    cts.randomize_demands = True
    cts.seed = None
    cts.anim = 2
    cts.l = 0.5 * np.sqrt(cts.wBlk**2 + cts.hBlk**2)
    cts.alpha = np.arctan(cts.wBlk / cts.hBlk)
    # design variables K (kN/m), Fult (kN), H (m)
    v = np.array([  500.0,     20.0,       3.0 ])  
    cost, constraints = analysis(v, cts)
    print(f'cost        = {cost:.6g} mÂ³')
    print(f'constraints = {constraints}')