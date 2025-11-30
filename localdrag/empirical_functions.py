#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: David Krach 
         david.krach@mib.uni-stuttgart.de

Copyright 2024 David Krach, Felix Weinhardt

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.

"""

### HEADER ------------------------------------------------------------------------

import numpy as np
import os

###--------------------------------------------------------------------------------

def lambda_wh(ratio, solver):
    """
    
    Parameters
    ----------
    ratio : float
        b/l to h ratio.
    
    Returns
    -------
    Prefactor lambda_wh dependend on the mean channel ratio.
    Float [ - ] 

    """
    
    if solver == 'stokes' or solver == 'brinkman':
        return 12. + np.divide(16., (np.exp(2.*1.36*ratio) - 1.), out=np.zeros_like(16.), where=(np.exp(2.*1.36*ratio) - 1)!=0)

    elif solver == 'analytical_brinkman':
        return 12. + np.divide(13.82642817, (np.exp(2.*13.02429246*ratio) - 1.), out=np.zeros_like(13.82642817), where=(np.exp(2.*13.02429246*ratio) - 1)!=0)
    elif solver == 'height_averaged':
        return 1.
    else:
        raise ValueError('Solver not given correctly!')

def lambda_gi(gradient, solver):
    """
    
    Parameters
    ----------
    gradient : float
        central difference quotient of the h-values of the domain
    
    Returns
    -------
    Pre-factor lambda_gi dependend on the non-dimensional gray-value gradient of the domain.
    Float [ - ]
    
    """
    if solver == 'stokes' or solver == 'brinkman':
        return 12. + 2.05960697 * gradient**2 + 6.41712348 * gradient

    elif solver == 'analytical_brinkman':
        if gradient == 0.0:
            return 12.0
        elif gradient >= 8:
            return 12.0
        else:
            return 1.60265829 * gradient + 10.01907011
        # return 10.0951389 * np.exp(-14.84259206*gradient) + 1.88066528
    elif solver == 'height_averaged':
        return 1.
    else:
        raise ValueError('Solver not given correctly!')


def lambda_p(gradient):
    """
    
    Parameters
    ----------
    gradient : float
        central difference quotient of the h-values of the domain
    
    Returns
    -------
    Pre-factor lambda_p dependend on the non-dimensional gray-value gradient of the domain.
    Float [ - ]
    
    """
    if gradient == 0.0:
        return 0.0
    else:
        return 1.14764447 * np.exp( -5.84167621 * gradient)
