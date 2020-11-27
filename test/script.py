"""Script to test and make the periakis algotithm work

I want to apply it to the result of an emcee exploration so I will use the simple example
of fitting a line provided in emcee tutorial (https://emcee.readthedocs.io/en/stable/tutorials/line/)

Susana's script (tools.compare) call a function tools.emcee_perrakis that is doing the job.
I can use it as is, because it requires the sampler to be created with blobs in a specific way.
This function calls tools.emcee_flatten that seems to format the chain and then perrakis.make_marginal_samples
that produces an input which is passed to perrakis.compute_perrakis_estimate that computes the log(Z).
This last function requires the lnlikelihood and lnposterior function and their arguments.
Finally you need the number of observations.

tools.emcee_flatten remove the burnin if provided and get only the chains that you want. Then it flattens the chains
to have one chain per parameter and then shuffle it. This function could be used as is but I have a function
to do that too in lisa (get_clean_flatchain) that is more flexible and then I jsut neeed to do np.random.shuffle.
The docstring of tools.emcee_flatten is a bit confusing when sampler is an iterable, because it asks for
data that are not needed by the function (lnprobability, acceptance, etc). Only the chain is needed, so
it must be tailored to some specific data format.

perrakis.make_marginal_samples seems to take the last nsamples from the flatchain and shuffle the samples
independantly for each parameter chain. I can use that as is.

perrakis.compute_perrakis_estimate computes the perrakis estimate of the Bayesian evidence. I can also
use it as is. I just need to provide the lnlikefunc and lnpriorfunc. I slightly modified compute_perrakis_estimate
so that he can apply the lnlikefunc and lnpriorfunc to a flatchain using map

"""
import sys
path_lisa = "/Users/olivier/Softwares/lisa/"
if path_lisa not in sys.path:
    sys.path.append(path_lisa)
from lisa.emcee_tools.emcee_tools import get_clean_flatchain
from lisa.tools.chain_interpreter import ChainsInterpret

import numpy as np
import matplotlib.pyplot as plt
import emcee

from perrakis import make_marginal_samples, compute_perrakis_estimate

np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
x0 = np.linspace(0, 10, 500)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

from scipy.optimize import minimize

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([m_true, b_true, np.log(f_true)]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
m_ml, b_ml, log_f_ml = soln.x

print("Maximum likelihood estimates:")
print("m = {0:.3f}".format(m_ml))
print("b = {0:.3f}".format(b_ml))
print("f = {0:.3f}".format(np.exp(log_f_ml)))

plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, np.dot(np.vander(x0, 2), w), "--k", label="LS")
plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_ml]), ":k", label="ML")
plt.legend(fontsize=14)
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

pos = soln.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)

plt.show()

nobs = len(x)

ntrys = 50  # 400
result = np.zeros(ntrys)
nsamples = 500

for ii in range(ntrys):
    fc = get_clean_flatchain(ChainsInterpret(sampler.chain, ['m', 'b', 'log_f']), l_walker=None, l_burnin=None, l_param_idx=None, force_finite=True)
    np.random.shuffle(fc)

    marginal = make_marginal_samples(fc, nsamples)

    ln_z = compute_perrakis_estimate(marginal_sample=marginal,
                                     lnlikefunc=log_likelihood, lnpriorfunc=log_prior,
                                     lnlikeargs=(x, y, yerr), lnpriorargs=(),
                                     densityestimation='histogram')
    ln_z += -0.5 * nobs * np.log(2 * np.pi)

    result[ii] = ln_z

print(f"DONE! Results: {result}")
