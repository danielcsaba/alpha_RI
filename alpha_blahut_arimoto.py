# ==================================================================
# Computing RI models with Sibson's
# information radius as cost function
# ---------------------------------------
#
# For details of implementation see "Attention Elasticities and 
# Invariant Information Costs" (Csaba, 2021)
# ==================================================================


import numpy as np
from scipy.spatial.distance import euclidean


class alpha_BA:
    """
    Solve for optimal information structure
    in Rational Inattention models with alpha-mutual information
    (Sibson's information radius) cost function.
    """

    def __init__(self, U, k, mu, a):
        """
        :param U: utility function given by payoff matrix (statesXactions)
        :param k: multiplier on cost function
        :param p: prior probability over states
        :param a: alpha parameter of Renyi-divergence
    
        :return: Return object with optimal information structure.
        """
    
        self.U, self.k, self.mu, self.a = U, k, mu, a
        assert (self.a > 0) & (self.a != 1), """Param a must be positive and
                                                different from 1."""
        self.num_state = U.shape[0]
        self.num_action = U.shape[1]
        assert self.num_state == len(mu), """The dimensions of the prior
                                             probability vector, p, are not
                                             aligned with payoff matrix."""
    
        # Optimal solution to populate
        self.opt_q = None
        self.opt_exp = None
        self.opt_L = None


    def _barycenter(self, P):
        """
        Compute barycenter of experiment P under prior 
        and given value of alpha.
        """
        q = np.dot(self.mu, P**self.a)**(1/self.a)
        q /= np.sum(q)
        
        return q


    def _f(self, L):
        """
        Computes optimal twisting of barycenter for conditional probabilities.
        """
        return np.maximum(((self.U/self.k - (L/self.mu)[:, np.newaxis])*
                           ((self.a - 1)/self.a)), 0)


    def _opt_cond_exp(self, L, q):
        """Dual-ascent to get optimal experiment for given barycenter q."""
        base = self._f(L)
        # exponentiate and deal with division by zero
        exp_denom = np.power(base, 
                             np.full_like(base, (1/(self.a - 1))), 
                             out=np.full_like(base, 0), 
                             where=(base > 1e-09))
        
        C = np.sum(exp_denom**self.a * q * self.mu[:, np.newaxis])    
     
        return (exp_denom * q)/C if C != 0 else exp_denom


    def _primal_feas(self, L, q):
        """
        Computes dual variable, L, that makes primal feasibility to be 
        satisfied for given value of barycenter, q.
        """
        exp = self._opt_cond_exp(L, q)
    
        return np.sum(exp, axis=1) - 1

    
    def _jac(self, L, q):
        """Jacobian of primal feasibility with respect to L dual."""
    
        # optimal conditional experiment given L, q
        base = self._f(L)
        # exponentiate and deal with division by zero
        exp_denom = np.power(base, 
                             np.full_like(base, (1/(self.a - 1))), 
                             out=np.full_like(base, 0), 
                             where=(base > 1e-09))
        
        C = np.sum(exp_denom**self.a * q * self.mu[:, np.newaxis])    
    
        exp = (exp_denom * q)/C if C != 0 else exp_denom
        
        # marginalize and get outer-product
        exp_marg = np.sum(exp, axis=1)
        J = np.outer(exp_marg, exp_marg)
        
        # add diagonal
        # exponentiate and deal with division by zero
        base_pow = np.power(base, 
                            np.full_like(base, ((2 - self.a)/(self.a - 1))),
                            out=np.full_like(base, 0), 
                            where=(base > 1e-09))
        
        diag_numer = np.sum((base_pow * q / 
                            (self.a * self.mu[:, np.newaxis])), 
                            axis=1)
        diag = np.diag((diag_numer/C))
                         
        return J - diag
    
    
    def _newton_root(self, q, x0, tol=1e-12, maxiter=int(1e4)):
        """
        Multivariate root finding via Newton's method.
        Consider custom bounds on admissible solution derived
        from the decision problem.
        """
        
        if self.a < 1:
            L_min = np.max(self.U * self.mu[:, np.newaxis] / self.k, axis=1)
        elif self.a > 1:
            L_max = np.max(self.U * self.mu[:, np.newaxis] / self.k, axis=1)
    
        t = 0
        dist = 100
        L = x0.copy()
        shrink = .75 # shrinkage to prevent overshooting
    
        while dist > tol and t < maxiter:
            # default Newton direction
            update_dir = np.linalg.inv(self._jac(L, q))@self._primal_feas(L, q)
            # deal with overshooting
            if self.a < 1:
                # step size to get to boundary with direction
                step_size_to_zero = np.divide((L - L_min), 
                                              update_dir, 
                                              out=np.ones_like(L_min), 
                                              where=(update_dir != 0))
                # correct stepsize if we would overshoot
                if np.max(step_size_to_zero) > 0:
                    _step_size = shrink*np.min(np.where(step_size_to_zero > 0, 
                                                        step_size_to_zero, 
                                                        np.inf))
                    step_size = np.minimum(1, _step_size)
                else:
                    step_size = 1
            elif self.a > 1:
                # step size to get to boundary with direction
                step_size_to_zero = np.divide((L - L_max), 
                                              update_dir, 
                                              out=np.ones_like(L_max), 
                                              where=(update_dir != 0))
                # correct stepsize if we would overshoot
                if np.max(step_size_to_zero) > 0:
                    _step_size = shrink*np.min(np.where(step_size_to_zero > 0, 
                                                        step_size_to_zero, 
                                                        np.inf))
                    step_size = np.minimum(1, _step_size)
                else:
                    step_size = 1            
            
            # update obeying bounds
            L_new = L - step_size*update_dir
            dist = euclidean(L, L_new)
            L  = L_new
            t += 1
    
        if t < maxiter:
            return L
        else:
            print('Root finding did not converge.')
            return None


    def opt_strat(self, q=None, tol=1e-9, maxiter=int(1e4), verbose=False):
        """Computes optimal barycenter and corresponding experiment."""
        
        # set q to uniform if not provided
        if q is None:
            q = np.ones(self.num_action)/self.num_action
        # set initial distance
        dist = 100

        # initial conditions for dual variables
        # make all probabilities interior at initial step
        if self.a > 1:
            L = np.min(self.U * self.mu[:, np.newaxis] / self.k, axis=1) - .2
        elif self.a < 1:
            L = np.max(self.U * self.mu[:, np.newaxis] / self.k, axis=1) + .2
        
        k = 0
        while dist > tol and k < maxiter:
            # ========================
            # P-step
            # ========================
            # Compute dual for given barycenter q
            L = self._newton_root(q=q, x0=L)
            # get optimal experiment given dual L and barycenter q
            P = self._opt_cond_exp(L, q)
            # ========================
            # q-step
            # ========================
            # Compute the updated barycenter
            q_new = self._barycenter(P)
            # Compute change in updated barycenter
            dist = euclidean(q, q_new)
            # update barycenter
            q = q_new
            k += 1
        
        if k < maxiter and verbose:
            print(f'Converged after {k} iterations.')
        elif k == maxiter:
            print('Exited without convergence.')
        
        self.opt_exp = P
        self.opt_q = q
        self.opt_dual = L
        
        return q, L, P


# -------------------------------
#    Class attributes starting
# -------------------------------

    np.set_printoptions(precision=8, suppress=True)

    @property
    def unconditional_prob(self):
        """Returns the unconditional choice probabilities."""
        return self.mu @ self.opt_exp

    @property
    def conditional_prob(self):
        """Returns the conditional choice probabilities
        i.e. the optimal experiment."""
        return self.opt_exp

    @property
    def opt_posterior(self):
        """Returns the optimal posteriors corresponding
        to each action."""
        return ((self.opt_exp * self.mu[:, np.newaxis]) / 
                (self.mu @ self.opt_exp))
