"""
Utility mathematical functions and common lineshapes for minimizer
"""
import numpy as np
from numpy import pi, log, exp, sqrt
from numpy.testing import assert_allclose

from scipy.special import gamma, gammaln, beta, betaln, erf, erfc, wofz

log2 = log(2)
sqrt2pi = sqrt(2*pi)

def gaussian(x, center=0.0, sigma=1.0):
    """Gaussian lineshape, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  variance

    Returns
    -------
    out :  1.0/(sigma*sqrt(2*pi)) exp(-(x - center)**2/(2*sigma**2))

    Notes
    -----
    FWHM = 2.354820 * sigma
    """
    return 1.0/(sigma*sqrt2pi) * exp(-(x-center)**2 /(2*sigma**2))


normalized_gaussian = gaussian

def lorentzian(x, center=0.0, sigma=1.0):
    """Lorentzian / Cauchy lineshape, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width

    Returns
    -------
    out :  (sigma/pi) / ((x-center)**2 + sigma**2)

    Notes
    -----
    FWHM = 2.0 * sigma
    """
    return (sigma/pi) / ((x-center)**2 + sigma**2)

def voigt(x, center=0.0, sigma=1.0, gamma=None):
    """Voigt function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter
    gamma :   float or None, default = None   width parameter

    Returns
    -------
    out :  Real[wofz(z) ] / (sigma*sqrt(2*pi))
           where wofz(z) is the fadeeva function (scipy.special.wofz)
           and z = (x-center + 1j*gamma)/ (sigma*sqrt(2))

    Notes
    -----
    when gamma is None, gamma is set = sigma
    
    FWHM = 3.60131 * sigma  when gamma = sigma

    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    if gamma is None:
        gamma = sigma

    z = (x-center + 1j*gamma)/ (sigma*sqrt(2))
    return wofz(z).real / (sigma*sqrt2pi)

def pvoigt(x, center=0.0, sigma=1.0, frac=0.5):
    """Pseudo-voigt function, 1 dimensional
    this returns a weighted sum of a Gaussian and Lorentzian function
    
    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter
    frac :    float, default = 0.5  lorentzian fraction

    Returns
    -------
    out : (1-frac)*gaussion(x, center, sigma) + frac*lorentzian(x, center, sigma)
    """
    gauss = gaussian(x, center, sigma)
    loren = lorentzian(x, center, sigma)
    return (1.0-frac)*gauss + frac*loren

def pearson7(x, center=0.0, sigma=1.0, expon=0.5):
    """pearson7 function, 1 dimensional

    using the definition from NIST StRD, though wikpedia differs.
    
    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter
    expon :   float, default = 0.5  exponent parameter

    Returns
    -------
    out : scale / (1 + ( ((1.0*x-center)/sigma)**2) * (2**(1/expon) -1) )**expon
          where
           scale = gamma(expon) * sqrt((2**(1/expon)-1)) / (gamma(expon-0.5) * sigma*sqrt(pi))
          (and gamma is the Gamma function, scipy.special.gamma)
    """
    scale = gamma(expon) * sqrt((2**(1/expon)-1)) / (gamma(expon-0.5) * sigma*sqrt(pi))
    return scale / (1 + ( ((1.0*x-center)/sigma)**2) * (2**(1/expon) -1) )**expon

def breit_wigner(x, center=0.0, sigma=1.0, q=1.0):
    """Breit-Wigner-Fano function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter
    q :       float, default = 1  q factor

    Returns
    -------
    out : (q*sigma/2 + x - center)**2 / ( (sigma/2)**2 + (x - center)**2 )
    """
    gam = sigma/2.0
    return  (q*gam + x - center)**2 / (gam*gam + (x-center)**2)

def damped_oscillator(x, center=1., sigma=0.1):
    """damped harmonic oscillator function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 1  centroid 
    sigma :   float, default = 0.1  width parameter

    Returns
    -------
    out : 1 /sqrt( (1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2))
    """
    cen = max(1.e-9, abs(cen))
    return (1./sqrt( (1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2))

def logistic(x, center=0., sigma=1.):
    """Logistic function, 1 dimensional
    
    yet another sigmoidal curve

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 1  centroid 
    sigma :   float, default = 0.1  width parameter

    Returns
    -------
    out : 1  - 1/(1 + exp((x-center)/sigma))
    """
    return (1. - 1/(1.0 + exp((x-center)/sigma)))

def lognormal(x, center=0., sigma=1.):
    """log-normal function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter

    Returns
    -------
    out : (1.0/(x*sigma*sqrt(2*pi))) * exp(-(log(x) - center)**2/ (2*sigma**2))
    """
    return (1.0/(x*sigma*sqrt2pi)) * exp(-(log(x) - center)**2/ (2*sigma**2))

def students_t(x, center=0.0, sigma=1):
    """Student's t distribution function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    sigma :   float, default = 1  width parameter

    Returns
    -------
    out :  gamma((sigma+1)/2) *(1 + (x-center)**2/sigma)^(-(sigma+1)/2)
           -------------------------
           sqrt(sigma*pi)*gamma(sigma/2)

    """
    expon = (sigma+1)/2.0
    denom = (sqrt(sigma*pi)*gamma(sigma/2))
    return (1 + (x-center)**2/sigma)**(-expon) * gamma(expon) / denom

def exponential(x, center=0., decay=1.0):
    """Exponential lineshape, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    decay :   float, default = 1  decay parameter

    Returns
    -------
    out :  exp(-(x-center)/decay)
    """
    return exp(-(x-center)/decay)


def powerlaw(x, center=0, expon=1):
    """Exponential lineshape, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    center :  float, default = 0  centroid 
    expon :   float, default = 1  exponent

    Returns
    -------
    out :  (x-center)**expon
    
    """
    return (x-center)**expon


def linear(x, offset=0.0, slope=0.0):
    """Linear function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    offset :  float, default = 0  offset (intercept)
    slope :   float, default = 0  slope


    Returns
    -------
    out :  offset + slope * x 
    
    """
    return offset + slope * x

def quadratic(x, offset=0.0, slope=0.0, quad=0.0):
    """quadratic function, 1 dimensional

    Parameters
    ----------
    x :       array_like          input values  
    offset :  float, default = 0  offset (intercept)
    slope :   float, default = 0  slope
    quad :    float, default = 0  quadratic


    Returns
    -------
    out :  offset + slope * x + qaud * x**2 
    
    """
    return offset + slope * x + quad * x**2

parabolic = quadratic


def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,
                         err_msg='', verbose=True):
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol, atol,
                        err_msg, verbose)
