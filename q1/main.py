#!/usr/bin/env python

from numpy import exp, dot, pi, inf, isclose, array, sum
from scipy.linalg import cholesky, LinAlgError, solve_triangular, inv
from scipy.integrate import nquad

# a
def integrand(v, a, w, p=0):
	v = array(v)
	return exp(-dot(v, dot(a, v))/2 + dot(v, w)) * (v**p).prod()

# Using Cholesky decomposition is a bit faster than using inv directly.
def true_integral(a, w):
	l = cholesky(a, lower=True)
	det_l = l.diagonal().prod()
	inv_l_w = solve_triangular(l, w, lower=True)
	return (2*pi)**(len(w)/2)/det_l*exp(dot(inv_l_w, inv_l_w)/2)

def num_integral(a, w):
	return nquad(
		lambda *args: integrand(args[:-2], *args[-2:]),
		[(-inf, inf)]*len(w),
		args=(a, w)
	)[0]

def verify(a, w):
	try:
		expected = true_integral(a, w)
		got = num_integral(a, w)
	except LinAlgError:
		print("error: not positive definite")
		return
	if isclose(expected, got):
		print("pass")
	else:
		print(f"fail: analytic {expected}, numerical {got}")

# b
A = [[4,2,1],[2,5,3],[1,3,6]]
W = [1,2,3]
#verify(A, W) # correct
#verify([[4,2,1],[2,1,3],[1,3,6]], W) # not positive definite

# c
# len(var) must be even number.
# For each possible way of pairing elements in var,
# multiply the corresponding elements in s.
# Then sum all the products.
# e.g. pairing(s, [1,2,3,4]) = s[1,2] s[3,4] + s[1,3] s[2,4] + s[1,4] s[2,3]
def pairing(s, var):
	order = len(var)
	if order == 0:
		return 1
	result = 0
	for i in range(1, order):
		result += s[var[0], var[i]] * pairing(s, var[1:i] + var[i+1:])
	return result

# For multivariate normal distribution with covariance matrix s and mean vector mu,
# calculate the moments specified by var.
# c specifies whether the corresponding element in var represents a variable shifted by its mean.
# e.g. var = [1,2,2] and c = [F,T,F] means the expectation value <v1 (v2-mu2) v2>.
def wick(s, mu, var, c=None):
	m = len(var)
	if m == 0:
		return 1
	if c is None:
		c = [False]*m
	result = 0
	for i in range(m):
		if c[i]:
			continue
		c[i] = True
		result += mu[var[i]] * wick(s, mu, var[:i] + var[i+1:], c[:i] + c[i+1:])
	if m % 2 == 0 and all(c):
		result += pairing(s, var)
	return result

def true_moment(a, w, p):
	s = inv(a)
	var = []
	for i, pi in enumerate(p):
		var += [i]*pi
	return wick(s, dot(s, w), var)

def num_moment(a, w, p):
	return nquad(
		lambda *args: integrand(args[:-3], *args[-3:]),
		[(-inf, inf)]*len(w),
		args=(a, w, p)
	)[0] / num_integral(a, w)
def verify_moment(a, w, p):
	expected = true_moment(a, w, p)
	got = num_moment(a, w, p)
	if isclose(expected, got):
		print("pass")
	else:
		print(f"fail: analytic {expected}, numerical {got}")

# In the comments, I give closed-form expressions for the moments.
# Below, S means A^-1, and mu means A^-1 w.
verify_moment(A, W, [1,0,0]) # mu1
verify_moment(A, W, [0,1,0]) # mu2
verify_moment(A, W, [0,0,1]) # mu3
verify_moment(A, W, [1,1,0]) # S12 + mu1 mu2
verify_moment(A, W, [0,1,1]) # S23 + mu2 mu3
verify_moment(A, W, [1,0,1]) # S13 + mu1 mu3
verify_moment(A, W, [2,1,0]) # 2 mu1 S12 + mu2 S11 + mu1^2 mu2
verify_moment(A, W, [0,1,2]) # 2 mu3 S23 + mu2 S33 + mu2 mu3^3
verify_moment(A, W, [2,2,0]) # (S11 + mu1^2) (S22 + mu2^2) + 2 S12^2
verify_moment(A, W, [0,2,2]) # (S22 + mu2^2) (S33 + mu3^2) + 2 S23^2
