# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:37:23 2017

@author: PJ
"""
import numpy as np
import pandas as pd
import copy
import sympy as spy
import scipy as scp
from scipy import stats

class cosinor_analysis_method(object):
    
    def __init__(self,user_id,data,w,alpha):
        
        self.data = data
        self.user_id = user_id
        self.t = np.array(self.data.index+1)/(float(len(self.data)))
        self.y = np.array(self.data.temperature)
        self.w = w
        self.alpha = alpha
        self.M = 0.
        self.Amp = 0.
        self.phi = 0.
        self.CI_M = 0.
        self.CI_phi_max = 0.
        self.CI_phi_min = 0.
        self.Amp_max = 0.
        self.Amp_min = 0.
        
        self.CI_Amp_max=0.
    
    #%% Parameter Estimation
    def parameter_estimation(self):
        
        self.n = len(self.t)
        
        # Substituition
        self.x = np.cos(self.w*self.t)
        self.z = np.sin(self.w*self.t)
        
        # Set up and solve the normal equations simultaneously
        NE = np.array([[self.n,np.sum(self.x),np.sum(self.z),np.sum(self.y)],
                        [np.sum(self.x),np.sum(self.x*self.x),np.sum(self.x*self.z),np.sum(self.x*self.y)],
                        [np.sum(self.z),np.sum(self.x*self.z),np.sum(self.z*self.z),np.sum(self.z*self.y)]])
        
        RNE_matrix = spy.Matrix(NE).rref()
        self.RNE = np.array(RNE_matrix[0]).astype(np.float64)
        self.M = self.RNE[0,3]
        self.beta = self.RNE[1,3]
        self.gamma = self.RNE[2,3]
        
        #Calculate amplitude and acrophase from beta and gamma
        self.Amp = np.sqrt(self.beta*self.beta + self.gamma*self.gamma)
        self.theta = np.arctan(abs(self.gamma/self.beta))
        
        #Calculate acrophase (phi) and convert from radians to degrees
        a = np.sign(self.beta);
        b = np.sign(self.gamma);
        if (a == 1 or a == 0) and b == 1:
            self.phi = -self.theta
        elif a == -1 and (b == 1 or b == 0):
            self.phi = -np.pi + self.theta
        elif (a == -1 or a == 0) and b == -1:
            self.phi = -np.pi - self.theta
        elif a == 1 and (b == -1 or b == 0):
            self.phi = -2*np.pi + self.theta
        
        self.f = self.M + self.Amp*np.cos(self.w*self.t+self.phi)
#        self.confidence_limtes_for_single_cosinor()
        #Display result
#        print 'Parameters:'+'-'*20
#        print 'Mesor = %f \nAmplitude = %f \nAcrophase = %f \n\n'%(self.M,self.Amp,self.phi)
        return self
    
    #%% confidence Limtes for Single Cosinor
    def confidence_limtes_for_single_cosinor(self):
        
        #Residual sum of errors
        self.RSS = np.sum(np.power((self.y - (self.M + self.beta*self.x + self.gamma*self.z)),2))
        
        #Residual varience estimation
        self.sigma = np.sqrt(self.RSS/np.float64(self.n-3))
        
        #Find confidence interval for mesor
        self.X = 1.0/self.n * np.sum(np.power((self.x - np.mean(self.x)),2))
        self.Z = 1.0/self.n * np.sum(np.power((self.z - np.mean(self.z)),2))
        self.T = 1.0/self.n * np.sum((self.x - np.mean(self.x))*(self.z - np.mean(self.z)))
        
        #Confidence interval for the mesor
        from scipy import stats
        self.CI_M = stats.t.ppf(1-self.alpha/2,self.n-3)*(self.sigma**2)*np.sqrt(((np.sum(self.x**2)*np.sum(self.z**2)) - (np.sum(self.x*self.z))**2)/(self.n**3*(self.X*self.Z - self.T**2)))
	
    #%%Confidence Interval Calculations
    def CIcalc(self):
        F_distr = stats.f.ppf(1-self.alpha,2,self.n-3)
        A=self.X
        B=2*self.T
        C = self.Z
        D = -2*self.X*self.beta - 2*self.T*self.gamma
        E = -2*self.T*self.beta - 2*self.Z*self.gamma
        F = self.X*self.beta**2 + 2*self.T*self.beta*self.gamma + self.Z*self.gamma**2 -(2.0/self.n)*self.sigma**2*F_distr
        g_max = -(2*A*E - D*B)/(4*A*C - B**2)
        gamma_s = np.arange(g_max-self.Amp*2,g_max+self.Amp*2+self.Amp/1000.0,self.Amp/1000.0)
        beta_s1 = (-(B*gamma_s + D) + np.sqrt((B*gamma_s + D)**2 - 4*A*(C*gamma_s**2 + E*gamma_s + F)+0j))/(2.0*A)
        beta_s2 = (-(B*gamma_s + D) - np.sqrt((B*gamma_s + D)**2 - 4*A*(C*gamma_s**2 + E*gamma_s + F)+0j))/(2.0*A)

        
        # Isolate ellipse region
        IND = beta_s1.real!=beta_s2.real
        gamma_s = gamma_s[IND]
        beta_s1 = beta_s1[IND]
        beta_s2 = beta_s2[IND]
        
        # Determine if confidence region overlaps the pole
        gamma_range = np.max(gamma_s)-np.min(gamma_s)
        if (gamma_range >=np.max(gamma_s)) and ((gamma_range>=np.max(beta_s1)) or (gamma_range>=np.max(beta_s2))):
            print('!! Confidence region overlaps the pole. Confidence limits for Amplitude and Acrophase cannot be determined !!')
            print(' ')
        
            self.CI_Amp_max = 0
            self.CI_Amp_min = 0
            self.CI_phi_max = 0
            self.CI_phi_min = 0
            
        else:
            # Confidence Intervals for Amplitude
            aaa = np.sqrt(beta_s1**2 + gamma_s**2)
            bbb = np.sqrt(beta_s2**2 + gamma_s**2)
            ccc = np.vstack((aaa,bbb))
            self.CI_Amp_max = np.max(np.max(ccc,axis=1))
            self.CI_Amp_min = np.min(np.min(ccc,axis=1))
            
            # Confidence Intervals for Acrophase
            dddd = np.arctan(np.abs(gamma_s/beta_s1))
            eeee = np.arctan(np.abs(gamma_s/beta_s2))
            theta = np.hstack((dddd,eeee))
            a = np.sign(np.hstack((beta_s1,beta_s2)))
            b = np.sign(np.hstack((gamma_s,gamma_s)))*3
            c = a + b 
            self.CIphi = np.zeros(len(c))
            for ii in range(len(c)):
                if c[ii]==4 or c[ii]==3:
                    self.CIphi[ii] = -theta[ii]
                    c[ii]= 1
                elif c[ii]==2 or c[ii]==-1:
                    self.CIphi[ii] = -np.pi + theta[ii]
                    c[ii]= 2
                elif c[ii]==-4 or c[ii]==-3:
                    self.CIphi[ii] = 3
                    c[ii] = 3
                elif c[ii] ==-2 or c[ii]==1:
                    self.CIphi[ii] = -2*np.pi+theta[ii]
                    c[ii] = 4
                    
            if np.max(c)-np.min(c)==3:
                self.CI_phi_max = np.min(self.CIphi[c==1])
                self.CI_phi_min = np.max(self.CIphi[c==4])
            else:
                self.CI_phi_max = np.max(self.CIphi)
                self.CI_phi_min = np.min(self.CIphi)
                
#            print self.CI_phi_max,self.CI_phi_min,self.CI_Amp_max,self.CI_Amp_min
            

    #%% Zero-amplitude test 
    def Zero_amplitude_test(self):	
        self.p_3a = stats.f.pdf((self.n*(self.X*self.beta**2 + 2*self.T*self.beta*self.gamma + self.Z*self.gamma**2)/(2*self.sigma**2)),2.0,self.n-3)
        
        
    def cosinor(self):
        '''Input:
            t - time series
        %   y - value of series at time t
        %   w - cycle length, defined by user based on prior knowledge of time
        %       series
        %   alpha - type I error used for cofidence interval calculations. Usually 
        %       set to be 0.05 which corresponds with 95% cofidence intervals
        %
        % Define Variables:
        %   M - Mesor, the average cylce value
        %   Amp - Amplitude, half the distance between peaks of the fitted
        %       waveform
        %   phi - Acrophase, time point in the cycle of highest amplitude (in
        %       radians)
        %   RSS - Residual Sum of Squares, a measure of the deviation of the
        %       cosinor fit from the original waveform
        %
        % Subfunctions:
        %   'CIcalc.m'
        %
        % Example:
        %   Define time series: 
        %       y = [102,96.8,97,92.5,95,93,99.4,99.8,105.5];
        %       t = [97,130,167.5,187.5,218,247.5,285,315,337.5]/360;
        %   Define cycle length and alpha:
        %       w = 2*pi;
        %       alpha = .05;
        %   Run Code:
        %       cosinor(t,y,w,alpha)
        '''
        if len(self.t)<4:
            print('There must be atleast four time measurements')
            return None
        
        self.parameter_estimation()
        self.confidence_limtes_for_single_cosinor()
        print('-'*30+'Parameters:'+'-'*30)
        print('Mesor = %f \nAmplitude = %f \nAcrophase = %f \n\n'%(self.M,self.Amp,self.phi))
        self.CIcalc()
        self.Zero_amplitude_test()
        print('Zero Amplitude Test')
        print('------------------------------------------------------')
        print('Amplitude        0.95 Confidence Limits        P Value')
        print('---------        ----------------------        -------')
        print(self.Amp,self.CI_Amp_min,self.CI_Amp_max,self.p_3a)
        print(' %.2f               (%.2f to %.2f)             %g'%(self.Amp,self.CI_Amp_min,self.CI_Amp_max,self.p_3a))
         
    def save_user_info(self):
        self.MSE= np.var(np.array(self.data.temperature)-self.f)
        self.high = max(self.f)
        self.low = min(self.f)
        self.high_time = self.data.measure_time[self.f==self.high].iloc[0]
        self.low_time = self.data.measure_time[self.f==self.low].iloc[0]
        self.metrics_dict = {'w':self.w,'alpha':self.alpha,
                             'M':self.M,'Amp':self.Amp,'phi':self.phi,
                             'high_time':self.high_time,'high':self.high,
                             'low_time':self.low_time,'low':self.low,
                             'CI_M':self.CI_M,'CI_Amp_min':self.CI_Amp_min,
                             'CI_Amp_max':self.CI_Amp_max,'CI_phi_min':self.CI_phi_min,
                             'CI_phi_max':self.CI_phi_max,'p_3a':self.p_3a,
                             'MSE':self.MSE}
        return self.metrics_dict
        
if __name__ == "__main__":
    
    #%%
#    file_path = '/home/peiji/company/psychology project/data/data_8_24_solved/test/'
#    file_name = '13566251628_solved.csv'
#    user_id = '13566251628_solved'
#    dt = pd.read_csv(file_path+file_name)
#    w = 2*np.pi
#    alpha = 0.05
#    a = cosinor_analysis_method(user_id,dt,w,alpha)
#    a.cosinor()
#    fit_dt = copy.deepcopy(dt)
#    fit_dt.temperature = a.f
#    user_lst = []
#    user_lst.append(dt)
#    user_lst.append(fit_dt)
#    import plot_tools as pto
#    pto.plot_data(user_id,user_lst)
    
    
    #%% test bug
   file_path = './'
   file_name = '13918950836.csv'
   user_id = '13918950836'
   dt = pd.read_csv(file_path+file_name)
   w = 2*np.pi
   alpha = 0.05
   a = cosinor_analysis_method(user_id,dt,w,alpha)
   a.cosinor()
   metrics_dt = a.save_user_info()
   fit_dt = copy.deepcopy(dt)
   fit_dt.temperature = a.f
   user_lst = []
   user_lst.append(dt)
   user_lst.append(fit_dt)
   import plot_tools as pto
   pto.plot_data(user_id,user_lst)
#    
    
    #%% batch 
#     w = 2*np.pi
#     alpha = 0.05
#     import os
#     FILE_DIR = '/home/peiji/company/psychology project/data/data_9_25/health/clean/'
#     FILELIST=os.listdir(FILE_DIR)
#     FILELIST = [FILELIST[i] for i in range(len(FILELIST))]
#     filelist = [x for x in FILELIST if x.endswith('.csv')]
#     ID = [x.split('.')[0] for x in filelist]
#     dict_temp ={}
#     metrics = []
# #    metrics_dict = {}
#     for user_id in ID:
#         print '-'*30+user_id+'-'*30
#         dt = pd.read_csv(FILE_DIR+user_id+'.csv')
#         if len(dt)==0: continue
#         a = cosinor_analysis_method(user_id,dt,w,alpha)
#         a.cosinor()
#
#         fit_dt = copy.deepcopy(dt)
#         fit_dt.temperature = a.f
#         user_lst = []
#         user_lst.append(dt)
#         user_lst.append(fit_dt)
#
#         # save plot
#         import plot_tools as pto
#         pto.plot_data(user_id,user_lst,fig_path='./healthy/figs/')
#
#         # save result
#         metrics_dict = a.save_user_info()
#         metrics.append(pd.DataFrame(metrics_dict,index=[user_id]))
#         metrics_dt = pd.concat(metrics)
#     result_path = './healthy/result/'
#     metrics_dt.to_csv(result_path+'metrics.csv')
#     metrics_dt.describe().to_csv(result_path+'metrics_statics_metrics.csv')
    
