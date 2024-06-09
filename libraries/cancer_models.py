import numpy as np
from scipy.integrate import odeint

#HYPERGENERIC PARAMETERS
eps = 0.01

#FUNCTION DEFINITIONS FOR THE CLASSICAL MODELS, FOR THE ODEINT 
def f_exp(x,t,r):
  return -r[0]*x

def f_log(x,t,r):
  f1 = r[0]*x*(1-(x/r[1])) # logistic
  return f1

def f_bert(x,t,r):
  f1 = r[0]*x**(2/3) - r[1]*x # Bertalanffy
  return f1

def f_gomp(x,t,r):
  f1 = r[0]*x*(r[1]-np.log(x)) # Gomperz
  return f1    

#INTEGRATE TO GET TRAJECTORIES
#Classical models
def model_1D(params,times,func,dt=0.1):
    time_span = np.arange(times[0],times[-1]+1,dt) #predict one day more than the last measurement

    y = odeint(func,y0=params[0],t=time_span,args=(params[1:],)).reshape(-1)
    return y
    
#Exponential stairs model as described in the thesis:
def model_exp_stairs(params,times,drug_times,dt=0.1):
    #constant:
    T = 2

    time_span = np.arange(times[0],times[-1]+1,dt)
    x = np.zeros(len(time_span))
    x[0] = params[0]
    tau = 0

    for t in range(1,len(time_span)):
      m = 0
      #when treatment is on, use -paramas[1]<0  <=> decay
      if t*dt - drug_times[tau] < T:
          m = -params[1]
      #when treatment is off, use paramas[1]<0  <=> growth
      else:
          m = params[2]

      #Euler:
      x[t] = x[t-1] + m * x[t-1]*dt

      if tau < len(drug_times) - 1:
          if drug_times[tau+1] < (t+1)*dt:
            tau = tau+1
    return x

def model_bert_stairs(params,times,drug_times,dt=0.1):
  #constants:
  T = 2


  time_span = np.arange(times[0],times[-1]+1,dt)
  x = np.zeros(len(time_span))
  x[0] = params[0]
  tau = 0
  
  for t in range(1,len(time_span)):
    #if treatment is on:
    if t*dt - drug_times[tau] < T and t*dt - drug_times[tau] > 0:
        a = 0
        b = params[2]+params[3]
    #if treatment is off:
    else:
       a = params[1]
       b = params[2]

    #Euler:
    x[t] = x[t-1] + dt* (a*x[t-1]**(2/3) - b*x[t-1]) # bertalanffy

    if tau < len(drug_times) - 1:
        if drug_times[tau+1] < (t+1)*dt:
          tau = tau+1
  return x

def model_SlS(pars, y_obs, max_time, drug_times,dt=0.1):
    #constants
    T = 2  #effect of drug in the body

    #parameters 
    S_0, K, delta_0, r_R_0, C, T = [pars[0], pars[1], pars[2], pars[3], pars[4], T]
    S_0 = S_0 * y_obs[0]
    K = K * y_obs[0]
    
    #initialize populations:
    N = int((max_time+1)/dt)
    S = np.zeros(N)
    S[0] = S_0
    R = np.zeros(N)
    R[0] = max(0, y_obs[0] - S[0])

    tau = 0 #variable to access drug_times
    r_S = 0 #growth_rate for the sensitive part
    for t in range(1,N):
      #when treatment is on
      if t*dt - drug_times[tau] < T:
        r_S = 0
        r_R = 0
        delta = delta_0
      #when treatment is off
      else:
        r_R = r_R_0
        r_S = 1
        delta = 0
    
      #Euler:
      S[t] = S[t-1] + dt * (r_S*(1 - (S[t-1] + R[t-1])/K ) * S[t-1] - delta * S[t-1])
      R[t] = R[t-1] + dt * (r_R *(1 - (C*S[t-1]+R[t-1])/K)*R[t-1]  - (delta/C) * R[t-1])

      #next drug_times
      if tau < len(drug_times) - 1:
        if drug_times[tau+1] < (t+1)*dt:
          tau = tau+1

    #total over all populations
    y = S + R
    return y

def model_SlS_star(pars, y_obs, max_time, drug_times, dt=0.1):
    T = 2
    K = 1.1*y_obs[0] 

    #parameters
    S_0,  delta_0, r, C= [pars[0], pars[1], pars[2], pars[3]]
    S_0 = S_0 * y_obs[0]
    
    N = int((max_time+1)/dt)
    S = np.zeros(N)
    S[0] = S_0
    R = np.zeros(N)
    R[0] = max(0, y_obs[0] - S[0])

    tau = 0
    r_S = 0
    for t in range(1,N):
      if t*dt - drug_times[tau] < T:
        r_S = 0
        r_R = 0
        delta = delta_0
      else:
        r_R = r/C
        r_S = r
        delta = 0

      S[t] = S[t-1] + dt * (r_S*(1 - (S[t-1] + R[t-1])/K ) * S[t-1] - delta * S[t-1])
      R[t] = R[t-1] + dt * (r_R *(1 - (C*S[t-1]+R[t-1])/K)*R[t-1]  - (delta/C) * R[t-1]) #the C in the logistic part is set to 3
     
      if tau < len(drug_times) - 1:
        if drug_times[tau+1] < (t+1)*dt:
          tau = tau+1
    y = S + R
    return y

def model_SlS_v2(pars, y_obs, max_time, drug_times, dt=0.1):
    #constants:
    T = 2
    K = 10**6 

    #parameters
    S_0,  delta_0, r, C, T = [pars[0], pars[1], pars[2], pars[3], T]
    S_0 = S_0 * y_obs[0]
    
    N = int((max_time+1)/dt)
    S = np.zeros(N)
    S[0] = S_0
    R = np.zeros(N)
    R[0] = max(0, y_obs[0] - S[0])

    tau = 0
    r_S = 0
    for t in range(1,N):
      if t*dt - drug_times[tau] < T:
        r_S = 0
        r_R = 0
        delta = delta_0
      else:
        r_R = r/C
        r_S = r
        delta = 0

      S[t] = S[t-1] + dt * (r_S*(1 - (S[t-1] + R[t-1])/K ) * S[t-1] - delta * S[t-1])
      R[t] = R[t-1] + dt * (r_R *(1 - (3*S[t-1]+R[t-1])/K)*R[t-1]  - (delta/C) * R[t-1]) #the C in the logistic part is set to 3
     
      if tau < len(drug_times) - 1:
        if drug_times[tau+1] < (t+1)*dt:
          tau = tau+1
    y = S + R
    return y
    
#Fast-Slow Exponential, this can work both with drug_times or with drug_times=None if no information is available
#- with parts=True it returns also the trajectories of the sub-populations
def model_fs_exp(params, y_obs, max_time, drug_times, dt=0.1, parts=False):
    #constant:
    T = 2

    #parameters
    S_0, death, C = params
    S_0 = S_0 * y_obs[0]

    #populations initialisation
    N = int((max_time+1)/dt)
    S = np.zeros(N)
    S[0] = S_0

    R = np.zeros(N)
    R[0] = max(0, y_obs[0] - S[0])

    tau = 0
    delta = death
    for t in range(1,N):
      #if information on the injections is available:
      if drug_times is not None:
        #if treatment is on
        if t*dt - drug_times[tau] < T:
          delta = death
        #if treatment is off
        else:
          delta = 0

      #Euler:
      S[t] = S[t-1] - dt * delta * S[t-1]
      R[t] = R[t-1] - dt * (delta/C) * R[t-1]

      if drug_times is not None:
        if tau < len(drug_times) - 1:
          if drug_times[tau+1] < (t+1)*dt:
            tau = tau+1
            
    y = S + R
    if parts:
      return y,S,R
    else:
      return y
    
#generic model wrapper
def model_wrapper(model_name,pars,y_obs,times,drug_times,dt=0.1):
    if model_name == "exp_stairs":
        y = model_exp_stairs(pars,times,drug_times,dt=dt)
    elif model_name == "bert_stairs":
        y = model_bert_stairs(pars, times, drug_times, dt=dt)
    elif model_name == "sls":
        y = model_SlS(pars,y_obs,times[-1],drug_times,dt=dt)
    elif model_name == "sls*":
        y = model_SlS_star(pars,y_obs,times[-1],drug_times,dt=dt)
    elif model_name == "slsv2":
        y = model_SlS_v2(pars, y_obs, times[-1], drug_times, dt=dt)
    elif model_name == "fs_exp":
        y = model_fs_exp(pars, y_obs, times[-1], drug_times, dt=dt)
    else:
        funcs_by_name = {"exp":f_exp, "log":f_log, "bert":f_bert, "gomp":f_gomp}
        y = model_1D(pars,times,funcs_by_name[model_name],dt=dt)
    return y
        
        
        

#GENERAL GAUSSIAN LOG-LIKELIHOOD
def log_likelihood(sigma,y_obs,y_pred,n=None):
  if n == None:
      n = len(y_obs)
  return 0.5*(n*np.log(2*np.pi) + n*np.log(sigma**2) + np.sum(((np.log((y_obs+eps)/(y_pred+eps)))**2)/(sigma**2)))
  

#OBJECTIVE FUNCTIONS WITH LOG-LIKELIHOOD
def objective_1D(params, times, y_obs,func,dt=0.1):
    y_pred = model_1D(params[1:],times,func,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs,y_pred)
    return l

def objective_exp_stairs(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_exp_stairs(params[1:],times,drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs,y_pred)
    return l

def objective_bert_stairs(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_bert_stairs(params[1:],times,drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs,y_pred)
    return l
    
def objective_SlS(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_SlS(params[1:],y_obs,times[-1],drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs, y_pred, n=len(y_obs)-1)
    return l

def objective_SlS_star(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_SlS_star(params[1:],y_obs,times[-1],drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs, y_pred, n=len(y_obs)-1)
    return l

def objective_SlS_v2(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_SlS_v2(params[1:],y_obs,times[-1],drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs, y_pred, n=len(y_obs)-1)
    return l
    
def objective_fs_exp(params, times, y_obs, drug_times,dt=0.1):
    y_pred = model_fs_exp(params[1:],y_obs,times[-1],drug_times,dt=dt)[np.array(times/dt).astype(int)]
    sigma = params[0]
    l = log_likelihood(sigma,y_obs, y_pred, n=len(y_obs)-1)
    return l
    
#generic objective wrapper:
def objective_wrapper(pars,model_name,times,y_obs,drug_times,dt=0.1):
    if model_name == "exp_stairs":
        l = objective_exp_stairs(pars,times,y_obs,drug_times,dt=dt)
    elif model_name == "bert_stairs":
        l = objective_bert_stairs(pars, times, y_obs, drug_times, dt=dt)
    elif model_name == "sls":
        l = objective_SlS(pars,times,y_obs,drug_times,dt=dt)
    elif model_name == "sls*":
        l = objective_SlS_star(pars,times,y_obs,drug_times,dt=dt)
    elif model_name == "slsv2":
        l = objective_SlS_v2(pars, times, y_obs, drug_times, dt=dt)
    elif model_name == "fs_exp":
        l = objective_fs_exp(pars, times, y_obs, drug_times, dt=dt)
    else:
        funcs_by_name = {"exp":f_exp, "log":f_log, "bert":f_bert, "gomp":f_gomp}
        l = objective_1D(pars,times,y_obs,funcs_by_name[model_name],dt=dt)
    return l
    
#generic objective wrapper without optimizing sigma (setting it to 1):
def objective_wrapper_no_sigma(pars,model_name,times,y_obs,drug_times,dt=0.1):
    pars = np.insert(pars, 0, 1)
    l = objective_wrapper(pars, model_name, times, y_obs, drug_times, dt=dt)
    return l
    
#SUPER Qs CONVERT THE NORMAL OBJECTIVE FUNCTIONS INTO A FORMAT COMPATIBLE WITH Pyswarm's PSO

def super_Q_wrapper(pars,model_name,times,y_obs,drug_times,dt=0.1):
    collector = []
    for p in pars:
        collector.append(objective_wrapper(p, model_name, times, y_obs, drug_times,dt=dt))
    return collector
    
#generic super-Q wrapper without optimizing sigma
def super_Q_wrapper_no_sigma(pars,model_name,times,y_obs,drug_times,dt=0.1):
    collector = []
    for p in pars:
        collector.append(objective_wrapper_no_sigma(p, model_name, times, y_obs, drug_times,dt=dt))
    return collector
    
    
#BOUNDS:
def get_bounds(model_name,y_obs_0=0):
  #model specific:
  model_bounds = {"exp":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,2)], 
                  "log":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,2), (0.001,5)], 
                  "bert":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,5), (0,5)], 
                  "gomp":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,1.2), (0,15)], 
                  "exp_stairs":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,5), (0,2)], 
                  "bert_stairs":[(0.01, 10), (y_obs_0*0.8, y_obs_0*1.2), (0,1), (10**-4,1), (0,5)],

                  "sls":[(0.01, 10), (0.85, 1), (1, 2), (1,10),(0.001,1),(1,30)],
                  "sls*":[(0.01, 10), (0.85, 1), (1,10),(0.001,1),(1,30)], 
                  "slsv2":[(0.01, 10), (0.85, 1), (0,10),(0,0.01),(1,30)], 
                  "fs_exp":[(0.01, 10), (0.92, 1),(0,5),(1,55)]}

  
  return np.array(model_bounds[model_name])
  