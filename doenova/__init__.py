# -*- coding: utf-8 -*-



#%%

import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
import pandas as pd
import statsmodels.api as sm
import pyDOE2 as DOE
import scipy.linalg
import warnings
warnings.filterwarnings("ignore")


GEN = [
[3,1,[[1,1]]],
[4,1,[[1,1,1]]],
[5,2,[[1,1,0],[1,0,1]]],
[5,1,[[1,1,1,1]]],
[6,3,[[1,1,0],[1,0,1],[0,1,1]]],
[6,2,[[1,1,1,0],[0,1,1,1]]],
[6,1,[[1,1,1,1,1]]],
[7,4,[[1,1,0],[1,0,1],[0,1,1]]],
[7,3,[[1,1,1,0],[0,1,1,1],[1,0,1,1]]],
[7,2,[[1,1,1,1,0],[1,1,1,0,1]]],
[7,1,[[1,1,1,1,1,1]]],
[8,4,[[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1]]],
[8,3,[[1,1,1,0,0],[1,1,0,1,0],[0,1,1,1,1]]],
[8,2,[[1,1,1,1,0,0],[1,1,0,0,1,1]]],
[8,1,[[1,1,1,1,1,1,1]]],
[9,5,[[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,1]]],
[9,4,[[1,1,1,1,0],[1,1,1,0,1],[1,1,0,1,1],[1,0,1,1,1]]],
[9,3,[[1,1,1,1,0,0],[1,0,1,0,1,1],[0,0,1,1,1,1]]],
[9,2,[[1,1,1,1,1,1,0],[1,1,1,0,1,1,1]]],
[9,1,[[1,1,1,1,1,1,1,1]]],
[10,6,[[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,1],[1,1,0,0]]],
[10,5,[[1,1,1,1,0],[1,1,1,0,1],[1,1,0,1,1],[1,0,1,1,1],[0,1,1,1,1]]],
[10,4,[[1,1,1,1,0,0],[1,1,1,0,1,0],[1,0,0,1,1,1],[0,1,0,1,1,1]]],
[10,3,[[1,1,1,0,0,0,1],[0,1,1,1,1,0,0],[1,0,1,1,0,1,0]]],
[10,2,[[1,1,1,1,1,1,0,0],[1,1,1,1,0,0,1,1]]],
[10,1,[[1,1,1,1,1,1,1,1,1]]],
[11,7,[[1,1,1,0],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,1],[1,1,0,0],[1,0,1,0]]],
[11,6,[[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[1,0,1,1,0],[1,0,0,1,1],[0,1,0,1,1]]],
[11,5,[[1,1,1,1,0,0],[1,1,1,0,1,0],[1,1,0,1,1,0],[1,0,1,1,1,1],[0,1,1,1,1,1]]],
[11,4,[[1,1,1,0,0,0,1],[0,1,1,1,1,0,0],[1,0,1,1,0,1,0],[1,1,1,1,1,1,1]]],
[11,3,[[1,1,1,1,1,1,0,0],[1,1,1,1,0,0,1,1],[1,1,0,1,0,1,1,1]]],
[11,2,[[1,1,1,1,1,1,0,0,0],[1,1,1,0,0,0,1,1,1]]],
[11,1,[[1,1,1,1,1,1,1,1,1,1]]],
[12,1,[[1,1,1,1,1,1,1,1,1,1,1]]],
[12,2,[[0,0,0,0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0,0,0]]],
[12,3,[[1,1,1,1,1,1,0,0,0],[1,1,1,1,0,0,1,1,0],[1,1,0,0,1,1,1,1,1]]],
[12,4,[[1,1,1,1,1,0,0,0],[1,1,1,0,0,1,1,0],[1,1,0,1,0,0,1,1],[1,0,1,0,1,0,1,1]]],
[12,5,[[1,1,1,0,0,0,0],[1,0,0,1,1,1,0],[0,1,0,1,1,0,1],[0,0,1,1,0,1,1],[1,1,1,0,1,1,1]]],
[12,6,[[1,1,1,0,0,0],[1,1,0,1,0,0],[1,0,1,1,1,0],[1,0,1,1,0,1],[1,1,0,0,1,1],[0,1,1,1,1,1]]],
[12,7,[[1,1,1,0,0],[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0],[1,1,0,0,1],[1,0,1,0,1],[1,0,0,1,1]]],
[12,8,[[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,1,1,1],[1,1,0,0],[1,0,1,0],[1,0,0,1]]],
]





#%%
# Unique source code

def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):

    ar = np.asanyarray(ar)
    if axis is None:
        return _unique1d(ar, return_index, return_inverse, return_counts)
    if not (-ar.ndim <= axis < ar.ndim):
        raise ValueError('Invalid axis kwarg specified for unique')

    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype
    # Must reshape to a contiguous 2D array for this to work...
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    if ar.dtype.char in (np.typecodes['AllInteger'] +
                         np.typecodes['Datetime'] + 'S'):
        # Optimization: Creating a view of your data with a np.void data type of
        # size the number of bytes in a full row. Handles any type where items
        # have a unique binary representation, i.e. 0 is only 0, not +0 and -0.
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts)
    if not (return_index or return_inverse or return_counts):
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return (uniq,) + output[1:]

def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.intp),)
            if return_inverse:
                ret += (np.empty(0, np.intp),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def intersect1d(ar1, ar2, assume_unique=False):

    if not assume_unique:
        # Might be faster than unique( intersect1d( ar1, ar2 ) )?
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]

def setxor1d(ar1, ar2, assume_unique=False):

    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = np.concatenate((ar1, ar2))
    if aux.size == 0:
        return aux

    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]


def in1d(ar1, ar2, assume_unique=False, invert=False):

    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject


    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)

    ar = np.concatenate((ar1, ar2))

    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):

    element = np.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


def union1d(ar1, ar2):

    return unique(np.concatenate((ar1, ar2)))

def setdiff1d(ar1, ar2, assume_unique=False):

    if assume_unique:
        ar1 = np.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
    


#%%


def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], int(m))
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:int(m),1:])
        for j in range(1, arrays[0].size):
            out[j*int(m):(j+1)*int(m),1:] = out[0:int(m),1:]
    return out



def create_req_matrix(req):
    
    if(False == isinstance(req, np.ndarray)):
        if (req==123456789):
            req = []
    
    required_interactions = req

    if(len(required_interactions)>0):
        required_interactions = np.array(required_interactions)*1
        if( required_interactions.ndim == 1 ):
            required_interactions = np.array([required_interactions])
        
    return required_interactions




def anovan(y,X,req=[],show_console=True):
    
    if(False == isinstance(req, np.ndarray)):
        if (req==123456789):
            req = []
    
    # Check if no replicates
    if( ( len(np.unique(X,axis=0)) == len(X) ) & (len(req)==0)  ):
        req = np.eye(len(np.array(X).T))
    
    required_interactions = create_req_matrix(req)
    
    y = np.array(y)
    if( y.ndim == 2 ):
        y = y[:,0]
    
    X = np.array(X) # Make sure it is a Numpy Array
    
    # Make X vertical
    if(X.ndim == 1):
        X = np.array([X]).T
    
    X_df = pd.DataFrame(X)  # Create a DataFrame
    for i in range(len(X.T)): # Rename Dataframe column names
        X_df = X_df.rename( columns={ i: ("f_"+chr(65+i)) } )
        
    X_df["y"] = y # Insert results into the dataframe

    k = [[0,1]] * len(X.T)
    all_comb = cartesian(k) # All posssible combinations

    # Create strings required to calculate Anovan models with the package statmodels
    all_strings = []
    for nbcomb in range(1,len(X.T)+1):
        string_t = ""
        count = 0
        for sel in all_comb:
            if(np.sum(sel) == nbcomb):

                if(count > 0):
                    string_t += " + "
                count += 1

                string_tt = ""
                countB = 0

                for i in range(len(sel)):
                    if(sel[i] == 1):

                        if(countB > 0):
                            string_tt+= "*"
                        countB += 1

                        string_tt += "C(f_"+chr(i+65)+")"

                string_t += string_tt

        all_strings.append(string_t)


    # Check the maximum number of interactions
    nb_inter = 0
    df_total = 0
    while( (df_total <= (len(X)-1)) & (nb_inter < len(X.T)) ):
        try:
            # Create the string for Statsmodels
            string_t = ""
            count = 0
            for sel in range(nb_inter+1):
                if(count > 0):
                    string_t += " + "
                count += 1
                string_t += all_strings[sel]

            # Calculate Anova
            model = ols('y ~ ' + string_t, X_df).fit()
            summary = sm.stats.anova_lm(model)

            df_total = np.sum(summary["df"])
            if(df_total <= (len(X)-1)):
                nb_inter += 1
        except:
            df_total = np.inf
    
    
    if (nb_inter > 0):
        
        # Create the string for statsmodels
        string_t = ""
        count = 0
        for sel in range(nb_inter):
            if(count > 0):
                string_t += " + "
            count += 1
            string_t += all_strings[sel]

        # Calculate Anova
        model = ols('y ~ ' + string_t, X_df).fit()
        summary = sm.stats.anova_lm(model)
        df_total = np.sum(summary["df"])


        # Calculate the total sum of squares
        sum_sq_total = np.sum( (y - np.mean(y))**2 )


        if (len(required_interactions)>0):

            summary_mod = pd.DataFrame(columns=['df','sum_sq','mean_sq','F','PR(>F)'])

            for sel in required_interactions:
                
                string_tt = ""
                string_tt2 = ""
                countB = 0

                # Create the string in order to select a row in the resulting DataFrame
                for i in range(len(sel)):
                    if(sel[i] == 1):

                        if(countB > 0):
                            string_tt+= ":"
                            string_tt2+= ""
                        countB += 1

                        string_tt += "C(f_"+chr(i+65)+")"
                        string_tt2 += chr(i+65)

                try: # Check if the row exists in the Dataframe
                    summary_mod.loc[string_tt2] = summary.loc[string_tt]
                except:
                    # If not, print it
                    print(string_tt2 + " ignored")

            # Updating residuals
            total_sq_sel = np.sum( summary_mod["sum_sq"] )
            total_df_sel = np.sum( summary_mod["df"] )
            residuals_sq_sel = sum_sq_total - total_sq_sel
            residuals_df_sel = len(X) - 1 - total_df_sel
            residuals_mean_squared = residuals_sq_sel/residuals_df_sel

            # Recalculate F and p values considering the updated residual components
            for i in range(len(summary_mod)):
                summary_mod.iloc[i,3] = summary_mod.iloc[i,2]/residuals_mean_squared
                summary_mod.iloc[i,4] = 1-st.f.cdf(summary_mod.iloc[i,3],summary_mod.iloc[i,0],residuals_df_sel)

            # Add the residual row
            summary_mod.loc["Err"] = [residuals_df_sel,residuals_sq_sel,residuals_mean_squared,"-","-"]

        else:

            # All interactions have to be presented
            summary_mod = summary
            all_indexes= summary_mod.index.tolist()
            
            # Replace index names
            for name in all_indexes:
                if (name != "Residual"):
                    newname = name.replace("C(f_", "")
                    newname = newname.replace(")", "")
                    newname = newname.replace(":", "")
                    #newname = ":" + newname
                    summary_mod = summary_mod.rename(index={ name:newname })

            # Rename the Residual index for Err
            summary_mod = summary_mod.rename(index={ "Residual":"Err" })
            summary_mod.iloc[-1,3:5] = "-" # Replace NaN with -
                
        summary_mod.loc["Tot"] = [len(X)-1,sum_sq_total,"-","-","-"]
        summary_mod = summary_mod.sort_index()  


        # Create the dictionary variable
        all_indexes = summary_mod.index.get_values()
        all_res = {}
        
        # Rename columns for Statgcb
        summary_mod = summary_mod.rename( columns={"df":"DF", "sum_sq":"SS", "mean_sq":"MS", "PR(>F)":"p"} )
        
        for i in all_indexes:
            
            if(summary_mod["DF"][i] == 0):
                print(i + " removed from ANOVA table")
                summary_mod = summary_mod.drop(i)
            else:
                h = { "DF":summary_mod["DF"][i], "SS":summary_mod["SS"][i], "MS":summary_mod["MS"][i], \
                      "F":summary_mod["F"][i], "p":summary_mod["p"][i]}
                name = i
                name = name.replace(":", "")
                all_res[name] = h
        
        if show_console:        
            print(summary_mod.to_string())
    
    else:
        
        print("Impossible to calculate an ANOVA with the provided data")
        summary_mod = []
        all_res = []
    
    
    
    return (required_interactions,all_res)





def make_2plan(nb_factors,nb_partial=0,nb_replicates=1,nb_center_points=0):

    nbvar = nb_factors - nb_partial
    all_choices = []
    for j in range(nbvar):
        all_choices.append([1,-1])
    allC = cartesian(all_choices)
    allC = allC.tolist()
    allCr = []
    for j in range(len(allC)):
        for k in range(nb_replicates):
            allCr.append(allC[j])
    
    # Add center points
    for _ in range(nb_center_points):
        allCr.append([0]*(nb_factors-nb_partial))
    
    # Select the right 
    if(nb_partial>0):
        cc= True
        count = 0
        while cc:
            if((GEN[count][0]==nb_factors) and (GEN[count][1]==nb_partial)):
                cc = False
                generators = GEN[count][2]
            else:
                count = count + 1
        
    
    # Partial
    pos = np.arange(0,nb_factors-nb_partial)
    allCrE = -np.array(allCr).astype(float)
    allCr= -np.array(allCr).astype(float)
    gen_string = ""
    for j in range(nb_partial):
        sel_gen = generators[j]
        
        # Show generator
        ss = chr(65+nbvar+j) + " = "
        for k in range(len(sel_gen)):
            if(sel_gen[k]==1):
                ss = ss + chr(65+k)
        gen_string = gen_string + "Generator "+ str(j+1) +": " + ss + "\n"
        FF = np.array(sel_gen)==1
        pos_sel = pos[FF]
        v = np.array([np.prod(allCr[:,pos_sel],axis=1)]).T
        allCrE = np.append(allCrE,v,axis=1)
        
    allCrEA = np.copy(allCrE)
    
    # Extend matrix
    inter = []
    for j in range(nb_factors):
        inter.append(chr(65+j))
    
    # Add partial choices
    for k in range(nb_partial):
        all_choices.append([1,-1])
        
    # Make 0-1 matrix
    allC = cartesian(all_choices)
    allC = (allC+1)/2
    
    allCs = np.sum(allC,axis=1).tolist()
    ii = sorted(range(len(allCs)), key=lambda k: allCs[k])
    allC = allC[ii,:]
    
    
    fc = 0
    if(nb_partial>0):
        fc = 0
    
    pos = np.arange(0,nb_factors)
    for j in range(allC.shape[0]):
        allC_sel = allC[j,:]
        if( (np.sum(allC_sel)>1) and (np.sum(allC_sel)<(nb_factors-nb_partial-fc+1)) ):
            FF = allC_sel==1
            posa = pos[FF]
            v = np.array([np.prod(allCrE[:,posa],axis=1)]).T
            allCrEA = np.append(allCrEA,v,axis=1)
            ssa = ""
            for l in range(len(allC_sel)):
                if(allC_sel[l]==1):
                    ssa = ssa + chr(l+65)
            inter.append(ssa)
    
    allCrEAu = unique(allCrEA,axis=1)
    
    ds = ""
    int_list = []
    for j in range(allCrEAu.shape[1]):
        count = 0
        sl = ""
        for k in range(allCrEA.shape[1]):
            if(np.array_equal(allCrEAu[:,j], allCrEA[:,k])):
                if(count>0):
                    ds = ds + "="
                    sl = sl + "="
                ds = ds + inter[k]
                sl = sl + inter[k]
                count = count + 1
        int_list.append([sl,allCrEAu[:,j]])
        ds = ds+ "\n"
            

    return allCr,allCrE,ds,gen_string,int_list






class expffplan:
    
    def __init__(self):
        self.Exp = []
        self.y = []
        self.blocks = 1
        self.specific_block = []
    
    
    def make_plan(self,nb_f,nb_replicates=1,nb_blocks=1):
        
        self.blocks = nb_blocks
        
        # If only one variable provided as an integer
        if(isinstance(nb_f, int)):
            nb_f = [nb_f]
        
        nb_f = list(nb_f)
        Exp_t = DOE.fullfact(nb_f)
        self.Exp = np.zeros([0,len(Exp_t.T)])
        
        for i in Exp_t:
            for rep in range(nb_replicates):
                self.Exp = np.append(self.Exp,np.array([i]),axis=0)
        return self.Exp
        
    
    def show_plan(self):
        if(len(self.Exp) > 0):
            print(str(self.Exp.shape[0]) + " runs:")
            print(self.Exp)
        else:
            print("NO EXISTING PLAN, please use make_plan")
        
        
           
    def export_plan(self,filename,random=False):
        if(len(self.Exp) > 0):
            data = self.Exp
            
            
            # Randomize if needed
            if(random | (self.blocks>1)):
                h = np.random.permutation(len(data))
                data = data[h]
            
            # Create the dataframe
            nb_col = len(data.T)
            data = pd.DataFrame(data)
            for i in range(nb_col):
                data = data.rename(columns={i:chr(65+i)})
            
            # Add the block column
            if(self.blocks>1):
                data["block"] = ""
                nb_obs_block = int(len(data)//self.blocks)
                for b in range(self.blocks):
                    data.iloc[b*nb_obs_block:(b+1)*nb_obs_block,-1] = b
                    
            
            # Add the results column
            data["result"] = ""
            
            w = pd.ExcelWriter(filename+".xlsx",engine='openpyxl')
            data.to_excel(w)
            w.save()
            
        
            
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
    
    
    def import_plan(self,filename):
        
        dataf = pd.read_excel(filename+".xlsx",0)
        data = np.array(dataf)
        
        # Number of blocks 
        if(dataf.columns.values.tolist()[-2] == "block"):
            nb_blocks = len(np.unique(dataf["block"]))
            ce = -2
            self.specific_block = data[:,-2]
        else:
            nb_blocks = 1
            ce = -1
        
        
        data_n0 = data[:,:ce]
        
        # Calculate the number of replicates        
        nb_var = len(data_n0.T)
        data_u = np.unique(data_n0,axis=0)
        nb_rep = int(len(data_n0)/len(data_u))
                        
        # Number of levels for each factor
        level_sel = []
        for i in range(nb_var):
            level_sel.append( len(np.unique(data_n0[:,i]) )  )
        
        _ = self.make_plan(level_sel,nb_rep,nb_blocks)
                
        # Modify data
        self.Exp = data[:,:ce]
        self.ExpE = data[:,:ce]
        
        # Insert the results if provided
        if(np.sum(dataf["result"])!=0):
            self.insert_results(np.array(dataf["result"]))
        
    
    
    def insert_results(self,y):
        if(len(self.Exp) > 0):
            if(isinstance(y, np.ndarray)):
                if(len(y)>1 and y.ndim>1):
                    y = y.T
                if(y.ndim > 1 ):
                    y = y.tolist()
                    y = y[0]
                else:
                    y = y.tolist()
            if(len(y) == self.Exp.shape[0]):
                self.y = y
                y = np.array(y)
            else:
                print("MISMATCHING DIMENSIONS BETWEEN X-y, no data inserted")
        else:
            print("NO EXISTING PLAN, please use make_plan")
        
    def anova(self,req=123456789, block=True):
        if( len(self.Exp) > 0 ):
            if( len(self.y) > 0):
                if(isinstance(req, np.ndarray)):
                    req = req.tolist()
                if(req != 123456789):
                    if(isinstance(req[0], int)):
                        req = [req]
                    req = np.array(req)
                    req = req[:,0:self.Exp.shape[1]]
                    req = req.tolist()
                X = self.Exp
                y = self.y
                #req,table_anova = anovan(y,X,req,)
                try:
                    req,table_anova = anovan0(y,X,req,\
                        blocks = self.blocks,sb = self.specific_block,\
                        nf=False, block_in = block)
                    self.req = req
                except:
                    table_anova = []
                    print("Impossible to perform the ANOVA")
                return table_anova
            else:
                print("No available results, cannot perform ANOVA")
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
            
            
    
    
    
    
    
    
class exppbplan:
    
    
    def __init__(self):
        self.Exp = []
        self.y = []
        self.center_points = 0
        self.blocks = 1
        self.specific_block = []
        self.ExpP = []
        self.level_values = []
        self.betas = []
        self.req = []
        
        
    def make_plan(self,nb_factors,nb_replicates=1,nb_cp=0,nb_blocks=1):
        
        nb_partial = 0
        nbvar = nb_factors - nb_partial
        all_choices = []
        for j in range(nbvar):
            all_choices.append([1,-1])
        
        # Create the design
        Exp_t = DOE.pbdesign(nb_factors-nb_partial)
        self.Exp = np.zeros([0,len(Exp_t.T)])
        for i in Exp_t:
            for rep in range(nb_replicates):
                self.Exp = np.append(self.Exp,np.array([i]),axis=0)
        
        self.center_points = nb_cp*nb_blocks
        self.blocks = nb_blocks
        
        if nb_cp>0:
            self.Exp = np.append(self.Exp,\
                    np.zeros([nb_blocks*nb_cp,len(self.Exp.T)]),axis=0)
        
        
        self.ExpE = self.Exp
        return self.ExpE
    
    
    def quantify_levels(self,vals):
        vals = np.array(vals)
        self.level_values = vals.copy()
        if(len(self.ExpE)>0):
            if( (self.ExpE.shape[1]==vals.shape[0]) and (2==vals.shape[1]) ):
                self.real_values = True
                A = np.copy(self.ExpE)
                for j in range(A.shape[1]):
                    # Calculate filters
                    ffm = A[:,j] == -1
                    ffp = A[:,j] == 1
                    ff0 = A[:,j] == 0
                    # Modify values
                    A[ffm,j] = vals[j,0]
                    A[ffp,j] = vals[j,1]
                    A[ff0,j] = np.mean(vals[j,:])
                self.ExpP = A
            else:
                print("Mismatching dimensions")
        else:
            print("NO EXISTING PLAN, please use make_plan")
        
        
    def show_plan(self,tt=0):
        if( len(self.ExpE) > 0 ):
            print(str(self.Exp.shape[0]) + " runs:")
            if(tt==1):
                print(self.ExpE)
            elif(tt==0):
                print(self.Exp)
        else:
            print("NO EXISTING PLAN, please use make_plan")
    
    
    def export_plan(self,filename,random=False):
        if(len(self.Exp) > 0):
            data = self.Exp
            
            # If levels were quantified, modify the plan
            if(len(self.level_values)>0): 
                data = self.ExpP.copy()
                center_points = np.array([np.mean(self.level_values,axis=1)])
            else:
                center_points = np.zeros([1,len(data.T)])
            
            # Randomize if needed
            if(random | (self.blocks>1)):
                if(self.blocks==1 | self.center_points==0): # If no blocks or no center points
                    h = np.random.permutation(len(data))
                    data = data[h]
                else: # Center point should be distributed
                    # Remove center points from blocks
                    f = np.prod(self.ExpE,axis=1) != 0
                    data_no0 = data[f]
                    # Permutate the data
                    h = np.random.permutation(len(data_no0))
                    data_no0 = data_no0[h]
                    # Center points per block
                    nb_center_block = int(self.center_points//self.blocks)
                    # Number of observations per block
                    nb_obs_block = int(len(data_no0)//self.blocks)
                    # Add center points in each block
                    all_blocks = []
                    for b in range(self.blocks): # For each block
                        data_sel = data_no0[b*nb_obs_block:(b+1)*nb_obs_block]
                        data_sel_o_size = len(data_sel)
                        for c in range(nb_center_block): # For each cp to add
                            pos = np.random.randint(0,data_sel_o_size+1)
                            if(pos==0):
                                data_sel = np.append(center_points,\
                                                     data_sel,axis=0)
                            elif(pos<len(data_sel)):
                                data_sel_p1 = np.append(data_sel[:pos],\
                                    center_points,axis=0)
                                data_sel = np.append(data_sel_p1,\
                                    data_sel[pos:],axis=0)
                            else:
                                data_sel = np.append(data_sel,\
                                        center_points,axis=0)
                        all_blocks.append(data_sel.copy())
                    # Create the new data
                    data = all_blocks[0]
                    for b in range(1,len(all_blocks)):
                        data = np.append(data,all_blocks[b],axis=0)
            
            nb_col = len(data.T)
            data = pd.DataFrame(data)
            for i in range(nb_col):
                data = data.rename(columns={i:chr(65+i)})
            
            
            # Add the block column
            if(self.blocks>1):
                data["block"] = ""
                nb_obs_block = int(len(data)//self.blocks)
                for b in range(self.blocks):
                    data.iloc[b*nb_obs_block:(b+1)*nb_obs_block,-1] = b
                    
            
            
            data["result"] = ""
            
            w = pd.ExcelWriter(filename+".xlsx",engine='openpyxl')
            data.to_excel(w)
            w.save()
            
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
    
    def import_plan(self,filename):
        
        dataf = pd.read_excel(filename+".xlsx",0)
        data = np.array(dataf)
        
        # Number of blocks 
        if(dataf.columns.values.tolist()[-2] == "block"):
            nb_blocks = len(np.unique(dataf["block"]))
            ce = -2
            self.specific_block = data[:,-2]
        else:
            nb_blocks = 1
            ce = -1
            
        # Modify level values to -1,0,1 if they were changed
        all_values = []
        ql = False
        for i in range(len(data.T)+ce):
            min_value = np.min(data[:,i])
            max_value = np.max(data[:,i])
            all_values.append([min_value,max_value])
            ff_min = data[:,i] == min_value
            ff_max = data[:,i] == max_value
            ff_mean = data[:,i] == np.mean([min_value,max_value])
            data[ff_min,i] = -1
            data[ff_max,i] = 1
            data[ff_mean,i] = 0
            if( (min_value != -1) | (max_value != 1) ):
                ql = True
        
        # Center points
        data_prod = np.prod(data[:,:ce],axis=1) != 0
        nb_ctr_points = len(data_prod) - np.sum(data_prod)
        
        data_n0 = data[data_prod,:-1]
        
        # Calculate the number of replicates        
        nb_var = len(data_n0.T)
        data_u = np.unique(data_n0,axis=0)
        nb_rep = int(len(data_n0)/len(data_u))
        
        _ = self.make_plan(nb_var,nb_rep,nb_ctr_points,nb_blocks)
                
        # Modify data
        self.Exp = data[:,:ce]
        self.ExpE = data[:,:ce]
        
        # Insert the results if provided
        if(np.sum(dataf["result"])!=0):
            self.insert_results(np.array(dataf["result"]))
            
        if ql: # Modify quantity levels if they were modified
            self.quantify_levels(np.array(all_values))
    
    
    def insert_results(self,y):
        if(len(self.Exp) > 0):
            if(isinstance(y, np.ndarray)):
                if(len(y)>1 and y.ndim>1):
                    y = y.T
                if(y.ndim > 1 ):
                    y = y.tolist()
                    y = y[0]
                else:
                    y = y.tolist()
            if(len(y) == self.Exp.shape[0]):
                self.y = y
                y = np.array(y)
            else:
                print("MISMATCHING DIMENSIONS BETWEEN X-y, no data inserted")
        else:
            print("NO EXISTING PLAN, please use make_plan")
        
        
        
    def anova(self,req=123456789, curv=True, block=True):
        if( len(self.Exp) > 0 ):
            if( len(self.y) > 0):
                if(isinstance(req, np.ndarray)):
                    req = req.tolist()
                if(req != 123456789):
                    if(isinstance(req[0], int)):
                        req = [req]
                    req = np.array(req)
                    req = req[:,0:self.Exp.shape[1]]
                    req = req.tolist()
                # Call the ANOVAn function
                try:
                    req,table_anova = anovan0(self.y,self.ExpE,req,\
                                blocks = self.blocks,sb = \
                                self.specific_block, curv_in=curv, block_in=block)
                    self.req = req
                except:
                    table_anova = []
                    print("Impossible to perform the ANOVA")
                if(isinstance(req, np.ndarray)):
                    _ = self.make_regression(req)
                else:
                    _ = self.make_regression()
                return table_anova
            else:
                print("No available results, cannot perform ANOVA")
        else:
            print("NO EXISTING PLAN, please use make_plan")
          
            
    def show_model(self):
        print("")
        print("y =~ ")
        if(len(self.betas)>0):
            print(self.betas[0,0])
        ss = ""
        for i in range(1,len(self.betas)):
            ff = self.req[i-1,:] == 1
            ss = ""
            for j in range(len(self.req.T)): 
                if(ff[j]):
                    ss += chr(j+65)
            print("+ " +  str(self.betas[i,0]) + " * " + ss)
            
            
    def make_regression(self,req=[]):
        
        if(len(req)): # If provided, create the requirements matrix
            self.req = create_req_matrix(req)
        
        if(len(self.req) == 0): # If no requirement matrix, use 1-order interactions
            self.req = np.eye(self.ExpE.shape[1])
            
        Xm = np.array([np.ones(self.ExpE.shape[0]).tolist()]).T
        pos = np.arange(0,self.ExpE.shape[1])
        for j in range(len(self.req)):
            FF = np.array(self.req[j])==1
            if(np.sum(FF)>0): # Make sure there is at least one var. selected
                pos_sel = pos[FF]
                v = np.array([np.prod(self.ExpE[:,pos_sel],axis=1)]).T
                Xm = np.append(Xm,v,axis=1)
        if(np.unique(Xm,axis=1).shape[1] != Xm.shape[1]):
            print("ERROR: redondant columns")
        else:
            if(len(self.y)>0):                
                y = np.array([self.y]).T
                b = np.linalg.inv(Xm.T@Xm)@Xm.T@y
                self.betas = b
                self.regress_model = sm.OLS(y,Xm)
                self.regress_model = self.regress_model.fit()
                #predictions = Xm@b
                return b,Xm
            else:
                return Xm


    def show_regress_stats(self):
        if(len(self.y)>0):  
            print(self.regress_model.summary())
            
            
            
            
    def predict_from_model(self,x_data=[]):
        
        if(len(x_data)>0): # If data are provided
            # Transform them
            x_data = np.array(x_data)
            if( x_data.ndim == 1 ):
                x_data = np.array([x_data])
        
            if( len(self.level_values)>0 ): # If levels were quantified
                for i in range(len(self.level_values)):
                    x_data[:,i] = 2*(x_data[:,i]-self.level_values[i,0])/\
                        (self.level_values[i,1]-self.level_values[i,0])-1     
        else: # If no data, use the plan
            x_data = np.array(self.ExpE)
        
        
        if(len(self.betas)>0):
            
            if( self.ExpE.shape[1] == len(x_data.T) ):
                
                Xm = np.array([np.ones(len(x_data)).tolist()]).T
                pos = np.arange(0,self.ExpE.shape[1])
                for j in range(len(self.req)):
                    FF = np.array(self.req[j])==1
                    if(np.sum(FF)>0): # Make sure there is at least one var. selected
                        pos_sel = pos[FF]
                        v = np.array([np.prod(x_data[:,pos_sel],axis=1)]).T
                        Xm = np.append(Xm,v,axis=1)

                y_pred = Xm@self.betas
                return y_pred[:,0].tolist()
                
            else:
                print("Dimensions mismatching")
            
        else:
            print("Please, create a model prior predicting")
                
    
    
    
    
    
    
    
    
    
    
    

class exp2plan:
    
    def __init__(self):
        self.Exp = []
        self.ExpE = []
        self.ExpP = []
        self.y = []
        self.int_list = []
        self.effects = []
        self.results = []
        self.inter = []
        self.req = []
        self.generators = []
        self.real_values = False
        self.center_points = 0
        self.blocks = 1
        self.specific_block = []
        self.betas = []
        self.level_values = []
    
    def make_plan(self,nb_factors,nb_partial=0,nb_replicates=1,nb_center_points=0,nb_blocks=1):
        if(nb_factors>nb_partial):
            infoexp = make_2plan(nb_factors,nb_partial,nb_replicates,nb_center_points*nb_blocks)
            if( len(infoexp[0])%nb_blocks == 0 ):
                self.Exp = infoexp[0]
                self.ExpE = infoexp[1]
                self.inter = infoexp[2]
                self.generators = infoexp[3]
                self.int_list = infoexp[4]
                self.center_points = nb_center_points*nb_blocks
                self.blocks = nb_blocks
                return self.ExpE
            else:
                print("Impossible number of blocks")
        else:
            print("Not valid")
            return []
    
    def quantify_levels(self,vals):
        vals = np.array(vals)
        self.level_values = vals.copy()
        if(self.inter != []):
            if( (self.ExpE.shape[1]==vals.shape[0]) and (2==vals.shape[1]) ):
                self.real_values = True
                A = np.copy(self.ExpE)
                for j in range(A.shape[1]):
                    # Calculate filters
                    ffm = A[:,j] == -1
                    ffp = A[:,j] == 1
                    ff0 = A[:,j] == 0
                    # Modify values
                    A[ffm,j] = vals[j,0]
                    A[ffp,j] = vals[j,1]
                    A[ff0,j] = np.mean(vals[j,:])
                self.ExpP = A
            else:
                print("Mismatching dimensions")
        else:
            print("NO EXISTING PLAN, please use make_plan")
    
    def insert_results(self,y):
        if(len(self.inter) > 0):
            if(isinstance(y, np.ndarray)):
                if(len(y)>1 and y.ndim>1):
                    y = y.T
                if(y.ndim > 1 ):
                    y = y.tolist()
                    y = y[0]
                else:
                    y = y.tolist()
            if(len(y) == self.Exp.shape[0]):
                self.y = y
                y = np.array(y)
                A = [["intercept",np.mean(y)]]
                for j in range(len(self.int_list)):
                    h = self.int_list[j][1]
                    if((np.sum(h==1)>0) and (np.sum(h==-1)>0)):
                        A.append([self.int_list[j][0],(np.mean(y[h==1])-np.mean(y[h==-1]))/2])
                    else:
                        A.append([self.int_list[j][0],"intercept"])
                self.effects = A
            else:
                print("MISMATCHING DIMENSIONS BETWEEN X-y, no data inserted")
        else:
            print("NO EXISTING PLAN, please use make_plan")
    
    def anova(self,req=123456789, wg=False, curv=True, block=True):
        if(len(self.inter) > 0):
            if(len(self.y) > 0):
                if(isinstance(req, np.ndarray)):
                    req = req.tolist()
                if(req != 123456789):
                    if(isinstance(req[0], int)):
                        req = [req]
                    req = np.array(req)
                    req = req[:,0:self.ExpE.shape[1]]
                    req = req.tolist()
                if(wg):
                    X = self.Exp
                else:
                    X = self.ExpE
                
                # Call the ANOVAn function
                try:
                    req,table_anova = anovan0(self.y,X,req,\
                                blocks = self.blocks,sb = \
                                self.specific_block, curv_in=curv, block_in=block)
                    self.req = req
                except:
                    table_anova = []
                    print("Impossible to perform the ANOVA.")
                if(isinstance(req, np.ndarray)):
                    _ = self.make_regression(req)
                else:
                    _ = self.make_regression()
                return table_anova
            else:
                print("No available results, cannot perform ANOVA")
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
            
    def show_plan(self,tt=1):
        if( len(self.inter) > 0 ):
            print(str(self.Exp.shape[0]) + " runs:")
            if(tt==1):
                print(self.ExpE)
            elif(tt==0):
                print(self.Exp)
            elif(tt==2):
                if(self.real_values):
                    print(self.ExpP)
                else:
                    print(self.ExpE)
        else:
            print("NO EXISTING PLAN, please use make_plan")
         
            
            
    def export_plan(self,filename,random=False):
        
        if(len(self.Exp) > 0):
            data = self.ExpE
            
            # If levels were quantified, modify the plan
            if(len(self.level_values)>0): 
                data = self.ExpP.copy()
                center_points = np.array([np.mean(self.level_values,axis=1)])
            else:
                center_points = np.zeros([1,len(data.T)])
            
            # Randomize if needed
            if(random | (self.blocks>1)):
                if(self.blocks==1 | self.center_points==0): # If no blocks or no center points
                    h = np.random.permutation(len(data))
                    data = data[h]
                else: # Center point should be distributed
                    # Remove center points from blocks
                    f = np.prod(self.ExpE,axis=1) != 0
                    data_no0 = data[f]
                    # Permutate the data
                    h = np.random.permutation(len(data_no0))
                    data_no0 = data_no0[h]
                    # Center points per block
                    nb_center_block = int(self.center_points//self.blocks)
                    # Number of observations per block
                    nb_obs_block = int(len(data_no0)//self.blocks)
                    # Add center points in each block
                    all_blocks = []
                    for b in range(self.blocks): # For each block
                        data_sel = data_no0[b*nb_obs_block:(b+1)*nb_obs_block]
                        data_sel_o_size = len(data_sel)
                        for c in range(nb_center_block): # For each cp to add
                            pos = np.random.randint(0,data_sel_o_size+1)
                            if(pos==0):
                                data_sel = np.append(center_points,\
                                                     data_sel,axis=0)
                            elif(pos<len(data_sel)):
                                data_sel_p1 = np.append(data_sel[:pos],\
                                    center_points,axis=0)
                                data_sel = np.append(data_sel_p1,\
                                    data_sel[pos:],axis=0)
                            else:
                                data_sel = np.append(data_sel,\
                                        center_points,axis=0)
                        all_blocks.append(data_sel.copy())
                    # Create the new data
                    data = all_blocks[0]
                    for b in range(1,len(all_blocks)):
                        data = np.append(data,all_blocks[b],axis=0)
                                
            
            nb_col = len(data.T)
            data = pd.DataFrame(data)
            for i in range(nb_col):
                data = data.rename(columns={i:chr(65+i)})
            
            # Add the block column
            if(self.blocks>1):
                data["block"] = ""
                nb_obs_block = int(len(data)//self.blocks)
                for b in range(self.blocks):
                    data.iloc[b*nb_obs_block:(b+1)*nb_obs_block,-1] = b
                    
            
            # Add the results column
            data["result"] = ""
            
            w = pd.ExcelWriter(filename+".xlsx",engine='openpyxl')
            data.to_excel(w)
            w.save()
            
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
            
        
    
    def import_plan(self,filename):
        
        dataf = pd.read_excel(filename+".xlsx",0)
        data = np.array(dataf)
        
        # Number of blocks 
        if(dataf.columns.values.tolist()[-2] == "block"):
            nb_blocks = len(np.unique(dataf["block"]))
            ce = -2
            self.specific_block = data[:,-2]
        else:
            nb_blocks = 1
            ce = -1
            
        # Modify level values to -1,0,1 if they were changed
        all_values = []
        ql = False
        for i in range(len(data.T)+ce):
            min_value = np.min(data[:,i])
            max_value = np.max(data[:,i])
            all_values.append([min_value,max_value])
            ff_min = data[:,i] == min_value
            ff_max = data[:,i] == max_value
            ff_mean = data[:,i] == np.mean([min_value,max_value])
            data[ff_min,i] = -1
            data[ff_max,i] = 1
            data[ff_mean,i] = 0
            if( (min_value != -1) | (max_value != 1) ):
                ql = True
        
        
        # Center points
        data_prod = np.prod(data[:,:ce],axis=1) != 0
        nb_ctr_points = len(data_prod) - np.sum(data_prod)
        
        data_n0 = data[data_prod,:ce]
        
        # Calculate the number of replicates        
        nb_var = len(data_n0.T)
        data_u = np.unique(data_n0,axis=0)
        nb_rep = int(len(data_n0)/len(data_u))
                        
        # Calculate the number of generators
        nb_gen = 2**nb_var / len(data_u)
        nb_gen = int(np.log(nb_gen)/np.log(2))
        
        # Make plan
        _ = self.make_plan(nb_var,nb_gen,nb_rep,nb_ctr_points//nb_blocks,nb_blocks)
                
        # Modify data
        self.Exp = data[:,:ce-nb_gen]
        self.ExpE = data[:,:ce]
        
        
        # Insert the results if provided
        if(np.sum(dataf["result"])!=0):
            self.insert_results(np.array(dataf["result"]))
        
        if ql: # Modify quantity levels if they were modified
            self.quantify_levels(np.array(all_values))
        
        
            
    def show_interactions(self):
        if(self.inter != []):
            print(self.inter)
        else:
            print("NO EXISTING PLAN, please use make_plan")
    
    def show_generators(self):
        if(self.inter != []):
            if(len(self.generators) == 0):
                print("No generator, complete plan")
            else:
                print(self.generators)
        else:
            print("NO EXISTING PLAN, please use make_plan")
        
    def show_effects(self):
        if(self.y != []):
            A = self.effects
            for j in range(len(A)):
                print(A[j][0]+" : "+str(A[j][1]))
        else:
            print("No y values")
            
            
    def make_regression(self,req=[]):
        
        if(len(req)): # If provided, create the requirements matrix
            self.req = create_req_matrix(req)
        
        if(len(self.req) == 0): # If no requirement matrix, use 1-order interactions
            self.req = np.eye(self.ExpE.shape[1])
            
        Xm = np.array([np.ones(self.ExpE.shape[0]).tolist()]).T
        pos = np.arange(0,self.ExpE.shape[1])
        for j in range(len(self.req)):
            FF = np.array(self.req[j])==1
            if(np.sum(FF)>0): # Make sure there is at least one var. selected
                pos_sel = pos[FF]
                v = np.array([np.prod(self.ExpE[:,pos_sel],axis=1)]).T
                Xm = np.append(Xm,v,axis=1)
        if(np.unique(Xm,axis=1).shape[1] != Xm.shape[1]):
            print("ERROR: redondant columns")
        else:
            if(len(self.y)>0):                
                y = np.array([self.y]).T
                b = np.linalg.inv(Xm.T@Xm)@Xm.T@y
                self.betas = b
                self.regress_model = sm.OLS(y,Xm)
                self.regress_model = self.regress_model.fit()
                #predictions = Xm@b
                return b,Xm
            else:
                return Xm


    def show_regress_stats(self):
        if(len(self.y)>0):  
            print(self.regress_model.summary())
        

    def show_model(self):
        print("")
        print("y =~ ")
        if(len(self.betas)>0):
            print(self.betas[0,0])
        ss = ""
        for i in range(1,len(self.betas)):
            ff = self.req[i-1,:] == 1
            ss = ""
            for j in range(len(self.req.T)): 
                if(ff[j]):
                    ss += chr(j+65)
            print("+ " +  str(self.betas[i,0]) + " * " + ss)


    def predict_from_model(self,x_data=[]):
        
        if(len(x_data)>0): # If data are provided
            # Transform them
            x_data = np.array(x_data)
            if( x_data.ndim == 1 ):
                x_data = np.array([x_data])
        
            if( len(self.level_values)>0 ): # If levels were quantified
                for i in range(len(self.level_values)):
                    x_data[:,i] = 2*(x_data[:,i]-self.level_values[i,0])/\
                        (self.level_values[i,1]-self.level_values[i,0])-1   
        else: # If no data, use the plan
            x_data = np.array(self.ExpE)
        
        
        if(len(self.betas)>0):
            
            if( self.ExpE.shape[1] == len(x_data.T) ):
                
                Xm = np.array([np.ones(len(x_data)).tolist()]).T
                pos = np.arange(0,self.ExpE.shape[1])
                for j in range(len(self.req)):
                    FF = np.array(self.req[j])==1
                    if(np.sum(FF)>0): # Make sure there is at least one var. selected
                        pos_sel = pos[FF]
                        v = np.array([np.prod(x_data[:,pos_sel],axis=1)]).T
                        Xm = np.append(Xm,v,axis=1)

                y_pred = Xm@self.betas
                return y_pred[:,0].tolist()
                
            else:
                print("Dimensions mismatching")
            
        else:
            print("Please, create a model prior predicting")



def mlregress(y,X,constant=False):
    
    # Modify y if necessary
    if(isinstance(y, list)):
        y = np.array([y]).T
    elif(y.shape[0]==1):
        y = y.T
    
    # Check dimensions
    X =  np.array(X)
    if(X.shape[0]!=y.shape[0]):
        raise ValueError("Dimensions mismatch")
    
    # Add constant if required
    if(constant):
        oo = np.ones([X.shape[0],1])
        X = np.append(X,oo,axis=1)
        
  
    results = sm.OLS(y,X).fit()
    print(results.summary())
    
    b = np.linalg.inv(X.T@X)@X.T@y
    
    return b 
        


def ac(c1,c2):
    
    if(isinstance(c1, list)):
        c1 = np.array([c1]).T
    elif(len(c1.shape)==1):
        c1 = np.array([c1]).T
    
    if(isinstance(c2, list)):
        c2 = np.array([c2]).T
    elif(len(c2.shape)==1):
        c2 = np.array([c2]).T
    
    if(c1.shape[0]==c2.shape[0]):
        X = np.append(c1,c2,axis=1)
    elif(c1.shape[0]==c2.shape[1]):
        X = np.append(c1,c2.T,axis=1)
    elif(c1.shape[1]==c2.shape[0]):
        X = np.append(c1.T,c2,axis=1)    
    else:
        raise ValueError("Dimensions mismatch")
    return X

# More consistant function
exp2fplan = exp2plan




class rsplan:
    
    
    def __init__(self,typed,param1,param2=0,param3=0,param4=0,param5=0):
        
        self.results=  []
        
        if(typed == "bbdesign"):
            self.ExpE = DOE.bbdesign(param1,param2)
            if(param3>1): # Replicated
                Expt = np.zeros([0,len(self.ExpE.T)])
                for i in self.ExpE:
                    if(np.sum(np.abs(i))>0):
                        for rep in range(param3):
                            Expt = np.append(Expt,np.array([i]),axis=0)
                    else:
                        Expt = np.append(Expt,np.array([i]),axis=0)
                self.ExpE = Expt
            print("{:.0f} runs".format(len(self.ExpE)))
            print(self.ExpE)
        elif (typed == "ccdesign") :
            self.ExpE = DOE.ccdesign(param1,param2,param3,param4)
            if(param5>1): # Replicates
                Expt = np.zeros([0,len(self.ExpE.T)])
                for i in self.ExpE:
                    for rep in range(param5):
                        Expt = np.append(Expt,np.array([i]),axis=0)
                self.ExpE = Expt
            print("{:.0f} runs".format(len(self.ExpE)))
            print(self.ExpE)
        else:
            print("Not existing design type")
    
    
    def get_plan(self):
        
        return self.ExpE
            
    
    def insert_results(self,y):
        if(len(self.ExpE) > 0):
            if(isinstance(y, np.ndarray)):
                if(len(y)>1 and y.ndim>1):
                    y = y.T
                if(y.ndim > 1 ):
                    y = y.tolist()
                    y = y[0]
                else:
                    y = y.tolist()
            if(len(y) == self.ExpE.shape[0]):
                self.y = y
                y = np.array(y)
            else:
                print("MISMATCHING DIMENSIONS BETWEEN X-y, no data inserted")
        else:
            print("NO EXISTING PLAN, please use make_plan")
            
    
    def anova(self,req=123456789):
        if( len(self.ExpE) > 0 ):
            if( len(self.y) > 0):
                if(isinstance(req, np.ndarray)):
                    req = req.tolist()
                if(req != 123456789):
                    if(isinstance(req[0], int)):
                        req = [req]
                    req = np.array(req)
                    req = req[:,0:self.ExpE.shape[1]]
                    req = req.tolist()
                X = self.ExpE
                y = self.y
                req,table_anova = anovan(y,X,req)
                self.req = req
                return table_anova
            else:
                print("No available results, cannot perform ANOVA")
        else:
            print("NO EXISTING PLAN, please use make_plan")
    
    
    
    def create_data(self,data,all_combinations):
        
        datam = np.ones([len(data),1])
        
        for cc in all_combinations:
            
            ccc = np.unique(cc).tolist()
            datam = np.append(datam,np.array([np.prod(data[:,ccc],axis=1)]).T,axis=1)
            
        return datam
                        
            
    
    def calculate_surface(self,types):
        
        if( len(self.y) > 0):
            
            types = int(types)
            
            if( (types>0) & (types<4) ):
            
                pp = np.arange(0,len(self.ExpE.T))
                pp = pp.tolist()
                pp = [pp]*types
                np.array(pp,dtype=int)
                
                self.all_combs = cartesian(pp)
                
                Xregress = self.create_data(self.ExpE,self.all_combs)
                self.C,_,_,_ = scipy.linalg.lstsq(Xregress, self.y)
                
                return (Xregress@np.array([self.C]).T)[:,0]
                
            else:      
                print("Specify an appropriate order")
                  
        else:
            print("No data to fit. Please use insert_results")
        
    
    
    def predict_from_surface(self,X):
        
        X = np.array(X)
        
        if(X.ndim == 1):
            X = np.array([X])
            
        Xregress = self.create_data(X,self.all_combs)
        return (Xregress@np.array([self.C]).T)[:,0]
   
    
    
    
def anovan0(y,X,req=[],show_console=True,blocks=1,sb=[],nf=True,curv_in=True,block_in=True):
    
    # Transform data to Numpy array
    y = np.array(y)
    if( y.ndim == 2 ):
        y = y[:,0]
    X = np.array(X) # Make sure it is a Numpy Array
    
    
    # Calculate the anova without center points and blocks
    if(nf):
        f0 = np.prod(X,axis=1) != 0
    else:
        f0 = np.prod(X,axis=1)*0 == 0
    required_interactions,anova_table = anovan(y[f0],X[f0],req,show_console=False)
    
    
    
    # Add the center points error (ignored if not required)
    f0 = f0 == 0
    if ( (np.sum(f0)>0) & curv_in ): 
        anova_table["Err"]["SS"] +=  np.sum( (y[f0]-np.mean(y[f0]))**2 )
        anova_table["Err"]["DF"] += np.sum(f0)-1
        anova_table["Tot"]["SS"] = np.sum( (y-np.mean(y))**2 )
        anova_table["Tot"]["DF"] = len(y)-1
    
    
    # Add the blocks
    if( (blocks > 1) & block_in ):
        # Number of observations per block
        nb_obs_block = int(len(X)//blocks)
        
        SS_block = 0
        
        if(len(sb) == 0): # Same size blocks
            # Calculate the mean of each block
            for b in range(blocks):
                SS_block += ( np.mean(y) - np.mean( y[ nb_obs_block*b:nb_obs_block*(b+1) ]) )**2
            SS_block = SS_block *  nb_obs_block
        else: # Different size blocks
            for b in range(blocks):
                fb = sb==b
                SS_block += np.sum(fb) * ( np.mean(y) - np.mean( y[fb ]) )**2
           
        
        DF_block = blocks-1
        
        # Remove SS associated to Blocks in Error
        anova_table["Err"]["SS"] -= SS_block
        anova_table["Err"]["DF"] -= DF_block
        
        if(anova_table["Err"]["SS"] < 0):
            print("WARNING: Negative Error SS")
        
        # Insert this information into the dictionary variable
        anova_table["Blocks"] = {"DF":DF_block, "SS":SS_block , "MS": SS_block/DF_block}
        
    
    # Update the error MS
    anova_table["Err"]["MS"] = anova_table["Err"]["SS"]/anova_table["Err"]["DF"]
    

    
    # Update the Anova table for bloakcs
    if( ((np.sum(f0)>0)&curv_in) | ((blocks>1)&block_in) ):
        SS_total = anova_table["Err"]["SS"]
        for i in anova_table.keys():
            if( (i != "Err") & (i != "Tot") ):
                anova_table[i]["F"] = anova_table[i]["MS"]/anova_table["Err"]["MS"]
                anova_table[i]["p"] = 1 - st.f.cdf( anova_table[i]["F"], \
                           anova_table[i]["DF"],anova_table["Err"]["DF"] )
                SS_total += anova_table[i]["SS"]
      
    
    # Add the curvature into the Anova table      
    if((np.sum(f0)>0) & curv_in):
        anova_table["Curv"] = {"DF":1, "SS":anova_table["Tot"]["SS"]-SS_total, \
                   "MS":anova_table["Tot"]["SS"]-SS_total, \
                   "F":(anova_table["Tot"]["SS"]-SS_total)/anova_table["Err"]["MS"], \
                   "p":1-st.f.cdf( (anova_table["Tot"]["SS"]-\
                                    SS_total)/anova_table["Err"]["MS"],\
                                    1,anova_table["Err"]["DF"])}
    
    
    # Show the new table
    summary_mod = pd.DataFrame(columns=['df','sum_sq','mean_sq','F','PR(>F)'])
    for i in anova_table.keys():
        if( (i != "Err") & (i != "Tot") & (i != "Blocks") ):
            summary_mod.loc[i] = [anova_table[i]["DF"],anova_table[i]["SS"],\
                            anova_table[i]["MS"],anova_table[i]["F"],\
                            anova_table[i]["p"]]
    
    if( (np.sum(f0)>0) & curv_in ): # Add the curvature into the table
        summary_mod.loc["Curv"] = [anova_table["Curv"]["DF"],\
                        anova_table["Curv"]["SS"],\
                        anova_table["Curv"]["MS"],anova_table["Curv"]["F"],\
                        anova_table["Curv"]["p"]]
        
    if( (blocks>1) & block_in ): # Add the blocks into the table
        summary_mod.loc["Blocks"] = [anova_table["Blocks"]["DF"],\
                        anova_table["Blocks"]["SS"],\
                        anova_table["Blocks"]["MS"],"-","-"]
    
    # Add the error and total at the end of the table
    summary_mod.loc["Err"] = [anova_table["Err"]["DF"],\
                        anova_table["Err"]["SS"],\
                        anova_table["Err"]["MS"],"-","-"]
    summary_mod.loc["Tot"] = [anova_table["Tot"]["DF"],\
                        anova_table["Tot"]["SS"],"-","-","-"]
    
    if show_console:        
        print(summary_mod.to_string())
    
    return (required_interactions,anova_table)



# Definition of synonyms
exp2f = exp2fplan
expff = expffplan
exppb = exppbplan
doe2f = exp2fplan
doeff = expffplan
doepb = exppbplan
doe_2f = exp2fplan
doe_ff = expffplan
doe_pb = exppbplan

