import numpy as np
import sys
import os
import math
import pandas as pd
from scipy.special import softmax

from .parameters import load_parameters

class ModelWrapper:
    def __init__(self, factors, nstates, nactions,nfutures):
        self.factors = factors
        self.nstates = nstates
        self.nactions = nactions
        self.nfutures = nfutures
        

    def get_attributes(self, attribute_names=None):
        if attribute_names is None:
            # If no attribute_names specified, get all attributes
            attribute_names = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        else:
            # Filter the specified attribute names
            attribute_names = [attr for attr in attribute_names if hasattr(self, attr)]
        attributes={}
        for attr in attribute_names:
            data=getattr(self, attr, np.nan) 
            if isinstance(data, np.ndarray):
                attributes[attr] = data.tolist()
            else:
                attributes[attr] = data
        return attributes
    
    def preprocess_params(self, parameters,mappingParam,default_values,nsplits):
        paramDict = {}
        for prm in mappingParam:
            if isinstance(mappingParam[prm], list) and len(mappingParam[prm])==1:
                paramDict[prm] = [parameters[mappingParam[prm][0]] for i in range(nsplits)] 
            elif isinstance(mappingParam[prm], list) and len(mappingParam[prm])>1:
                paramDict[prm] = [parameters[mappingParam[prm][i]] for i in range(nsplits)]
            elif isinstance(mappingParam[prm], float):
                paramDict[prm] = [mappingParam[prm] for i in range(nsplits)]
            else:
                paramDict[prm] = [default_values[prm] for i in range(nsplits)]
                
        for prm in mappingParam:
            if isinstance(mappingParam[prm], str):
                paramDict[prm] = [paramDict[mappingParam[prm]][i] for i in range(nsplits)]
                
        return paramDict

        
    def init(self, extraParam):
        
        X0 = np.zeros((0,))
        mappingX = {}
        
        SAS = (1/self.nfutures)*np.ones((self.nstates, self.nactions,self.nfutures))
        mappingX['SAS'] = X0.size + np.arange(self.nstates * self.nactions * self.nfutures).reshape((self.nstates, self.nactions,self.nfutures))
        X0 = np.concatenate([X0, SAS.flatten()])

        SS = (1/self.nfutures)*np.ones((self.nstates,self.nfutures))
        mappingX['SS'] = X0.size + np.arange(self.nstates*self.nfutures).reshape((self.nstates,self.nfutures))
        X0 = np.concatenate([X0, SS.flatten()])
                
        mappingX['Omega'] = np.array([X0.size])
        X0 = np.concatenate([X0, np.array([0.0])])
        
        if 'forceArbitrator' in extraParam:
            mappingX['arbitrator'] = np.array([X0.size])
            X0 = np.concatenate([X0,np.array([extraParam['forceArbitrator']])])           
        else:
            mappingX['arbitrator'] = np.array([X0.size])
            X0 = np.concatenate([X0,np.array([0.75])])
                
        self.X0=X0
        self.mappingX=mappingX
    
    def compute_logpriors(self,parameters,prior_array,mappingParam):
        priorNLL=0.0
        for prm in mappingParam:
            if isinstance(mappingParam[prm], list):
                for param_ind in mappingParam[prm]:
                    priorNLL-=prior_array[param_ind].log_pdf(parameters[param_ind])[0]
                    
        return priorNLL
    
    def fit(self,
            parameters,
            mappingParam,
            arrayS,
            arrayA,
            arraySnext,
            arrayR,
            arrayType,
            arrayMissed,
            arrayPrediction,
            arraySplit,
            arrayRTs=None,
            resets=[],
            returnMemory=False,
            prior_array=None,
            default_values={}):

        if returnMemory:
            memory_snapshots = []

        X = np.copy(self.X0)
        predNLL = 0.0
        rtNLL = 0.0
        
        alternative_futures=[[1,2],[0,2],[0,1]]
        
        nsplits=np.unique(arraySplit).shape[0]
        self.paramDict=self.preprocess_params(parameters, mappingParam, default_values,nsplits)

        
        for step, state in enumerate(arrayS):
        
            # run reset
            if step in resets:
                X = np.copy(self.X0)
            
            #
            action=arrayA[step]
            next_state=arraySnext[step]
            split=arraySplit[step]

            # exploration trial (learning only)
            if arrayType[step]==0 and arrayMissed[step]==0 and next_state>=0:
                omegaPE=X[self.mappingX['SAS'][state,action,next_state]]-X[self.mappingX['SS'][state,next_state]]-X[self.mappingX['Omega'][0]]
                ssPE=1.0-X[self.mappingX['SS'][state,next_state]]
                sasPE=1.0-X[self.mappingX['SAS'][state,action,next_state]]
                #
                X[self.mappingX['Omega'][0]]+=self.paramDict['alphaOmega'][split]*omegaPE
                #
                X[self.mappingX['SS'][state,next_state]] += self.paramDict['alphaSS'][split]*ssPE
                X[self.mappingX['SS'][state,alternative_futures[next_state]]] *= (1.0-self.paramDict['alphaSS'][split])*X[self.mappingX['SS'][state,alternative_futures[next_state]]]
                X[self.mappingX['SS'][state, :]] /= X[self.mappingX['SS'][state, :]].sum()
                #
                X[self.mappingX['SAS'][state,action,next_state]] += self.paramDict['alphaSAS'][split]*sasPE
                X[self.mappingX['SAS'][state,action,alternative_futures[next_state]]] *= (1-0-self.paramDict['alphaSAS'][split])*X[self.mappingX['SAS'][state,action,alternative_futures[next_state]]]
                X[self.mappingX['SAS'][state,action,:]] /= X[self.mappingX['SAS'][state,action,:]].sum()
                
                if returnMemory:
                    
                        memory_data = np.concatenate([X.copy(), np.array([omegaPE,ssPE,sasPE,np.nan,np.nan])])
                        memory_snapshots.append(memory_data)
            
            # prediction trial (choice + some learning)
            elif arrayType[step]==1 and arrayMissed[step]==0 and arrayPrediction[step]>=0:
                                
                # arbitrator for this trial
                arbLogit = self.paramDict['slopeOmega'][split]*X[self.mappingX['Omega'][0]]
                arbitrator= 1 / (1 + math.exp(-(self.paramDict['biasArbitrator'][split] + arbLogit)))
                X[self.mappingX['arbitrator'][0]]=min(max(arbitrator, 10**-10), 1-(10**-10))      
                
                probSAS=X[self.mappingX['SAS'][state,action,:]]
                probSS=X[self.mappingX['SS'][state,:]]
                
                probMixed=probSAS*arbitrator+probSS*(1-arbitrator)
                logits =  (probMixed-np.max(probMixed))*self.paramDict['betaPred'][split]     
                                     
                predProb = (1.0/self.nfutures) * self.paramDict['epsilon'][split] + (1 - self.paramDict['epsilon'][split])*softmax(logits)
                predNLL -= math.log(predProb[arrayPrediction[step]])
                
                if arrayR[step]>=0:
                    
                    omegaPE=X[self.mappingX['SAS'][state,action,next_state]]-X[self.mappingX['SS'][state,next_state]]-X[self.mappingX['Omega'][0]]
                    ssPE=1.0-X[self.mappingX['SS'][state,next_state]]
                    sasPE=1.0-X[self.mappingX['SAS'][state,action,next_state]]
                    #
                    X[self.mappingX['Omega'][0]]+=self.paramDict['alphaOmega'][split]*omegaPE
                    #
                    X[self.mappingX['SS'][state,next_state]] += self.paramDict['alphaSS'][split]*ssPE
                    X[self.mappingX['SS'][state,alternative_futures[next_state]]] *= (1.0-self.paramDict['alphaSS'][split])*X[self.mappingX['SS'][state,alternative_futures[next_state]]]
                    X[self.mappingX['SS'][state, :]] /= X[self.mappingX['SS'][state, :]].sum()
                    #
                    X[self.mappingX['SAS'][state,action,next_state]] += self.paramDict['alphaSAS'][split]*sasPE
                    X[self.mappingX['SAS'][state,action,alternative_futures[next_state]]] *= (1-0-self.paramDict['alphaSAS'][split])*X[self.mappingX['SAS'][state,action,alternative_futures[next_state]]]
                    X[self.mappingX['SAS'][state,action,:]] /= X[self.mappingX['SAS'][state,action,:]].sum()

                
                    if returnMemory:
                        memory_data = np.concatenate([X.copy(),np.array([omegaPE,ssPE,sasPE,np.argmax(predProb),predProb[arrayPrediction[step]]])])
                        memory_snapshots.append(memory_data)
                else:
                    
                    if returnMemory:
                        memory_data = np.concatenate([X.copy(),np.array([np.nan,np.nan,np.nan,np.argmax(predProb),predProb[arrayPrediction[step]]])])
                        memory_snapshots.append(memory_data)
            
            else:
                
                if returnMemory:
                    memory_data = np.concatenate([X.copy(), np.array([np.nan,np.nan,np.nan,np.nan,np.nan])])
                    memory_snapshots.append(memory_data)
                                    
        if returnMemory:
            colnames=[]
            for key, array in self.mappingX.items():
                if array.size > 1:
                    for idx in np.ndindex(array.shape):
                        idx_str = ','.join(map(str, idx))
                        colnames.append(f"{key}({idx_str})")
                else:
                    colnames.append(key)
                    
            colnames.append('OmegaPE')
            colnames.append('SSPE')
            colnames.append('SASPE')
            colnames.append('predModelGuess')     
            colnames.append('predLikelihood')
            agentMemory = pd.DataFrame(memory_snapshots, columns=colnames) 
            

        if prior_array is not None:
            #priorNLL=0.0
            priorNLL=self.compute_logpriors(parameters,prior_array,mappingParam)
        else:
            priorNLL=0.0
            
        return (predNLL,priorNLL, agentMemory, self.mappingX) if returnMemory else predNLL+priorNLL

def setup_models(bounds_list, plausible_bounds_list, prior_shapes, default_values):

    """Define and load the model space."""
    model_list = []

    # Define models here
    all_models=[
        {
            'included': True,
            'agent': ModelWrapper,
            'factors': ['SAS', 'SS', 'Omega', 'arbitrator'],
            'name': 'SASSS_Omega_standard',
            'parameter_preset': {},
            'parameter_mapping':{'alphaSAS':[0], 'betaPred': [1], 'slopeOmega': [2],'biasArbitrator':[3], 'epsilon': 0.001, 
                                'alphaSS':'alphaSAS','alphaOmega':'alphaSAS'},
            'bounds_list':bounds_list,
            'plausible_bounds_list': plausible_bounds_list,
            'prior_shapes': prior_shapes,
            'default_values': default_values,
        },
        {
            'included': True,
            'agent': ModelWrapper,
            'factors': ['SAS', 'SS', 'Omega', 'arbitrator'],
            'name': 'SASSS_Omega_splitBias',
            'parameter_preset': {},
            'parameter_mapping':{'alphaSAS':[0], 'betaPred': [1], 'slopeOmega': [2],'biasArbitrator':[3,4], 'epsilon': 0.001, 
                                'alphaSS':'alphaSAS','alphaOmega':'alphaSAS'},
            'bounds_list':bounds_list,
            'plausible_bounds_list': plausible_bounds_list,
            'prior_shapes': prior_shapes,
            'default_values': default_values,
        },
        {
            'included': True,
            'agent': ModelWrapper,
            'factors': ['SAS', 'SS', 'Omega', 'arbitrator'],
            'name': 'SASSS_Omega_splitBeta',
            'parameter_preset': {},
            'parameter_mapping':{'alphaSAS':[0], 'betaPred': [1,2], 'slopeOmega': [3],'biasArbitrator':[4], 'epsilon': 0.001, 
                                'alphaSS':'alphaSAS','alphaOmega':'alphaSAS'},
            'bounds_list':bounds_list,
            'plausible_bounds_list': plausible_bounds_list,
            'prior_shapes': prior_shapes,
            'default_values': default_values,
        },
        {
            'included': True,
            'agent': ModelWrapper,
            'factors': ['SAS'],
            'name': 'SASonly',
            'parameter_preset': {'forceArbitrator': 1.0},
            'parameter_mapping':{'alphaSAS':[0], 'betaPred': [1], 'epsilon': 0.001,
                                'alphaSS':'alphaSAS','alphaOmega':'alphaSAS'},
            'bounds_list':bounds_list,
            'plausible_bounds_list': plausible_bounds_list,
            'prior_shapes': prior_shapes,
            'default_values': default_values,
        },
    ]
    for model in all_models:
        if model['included']:
            model_list.append(model)

    return model_list,all_models

def main():
    """Main execution function."""
    bounds_list, plausible_bounds_list, default_values, prior_shapes = load_parameters()

    model_list, all_models = setup_models(bounds_list, plausible_bounds_list, prior_shapes, default_values)

    print('Models loaded:')
    for m, model in enumerate(model_list):
        print(f"{m}. {model['name']}")
        
    return model_list, bounds_list, plausible_bounds_list, all_models, prior_shapes, default_values


if __name__ == "__main__":
    main()