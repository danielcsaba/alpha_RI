# ============================================================
#  Choice probabilities in Rational Inattention models
#  with Shannon (mutual information) cost functions
#  Daniel Csaba
#  February, 2017
# ============================================================



# ============================================================
# Solve for the optimal information structure
# using the Blahut-Arimoto algorith. 
# See Rate Distortion Function in Cover & Thomas (2006).
# ============================================================


import numpy as np
from scipy.spatial.distance import euclidean


class BA:
    """ Solve for optimal information structure
    in Rational Inattention models with Shannon
    cost function."""

    def __init__(self, U, k, mu):
        """
        :param U: utility function given by payoff matrix (statesXactions)
        :param k: multiplier on Shannon cost function
        :param p: prior probability over states
        
        :return: Return object with optimal information structure
                 for information processing.
        """
        
        self.U, self.k, self.mu = U, k, mu
        # Number of states
        self.num_state = U.shape[0]
        # Number of actions
        self.num_action = U.shape[1]
        # Check compatibility of U and p
        assert self.num_state == len(mu), """The dimensions of the prior 
                                             probability vector, p, are not 
                                             aligned with payoff matrix."""
        
        # optimal strategy to populate
        self.opt_q = None
        self.opt_exp = None 


    def opt_strat(self, q=None, tol=1e-12, maxiter=int(1e4), verbose=False):
        """Computes the optimal attention strategy."""
        # Initial barycenter -- unconditional choice prob
        if q is None:
            q = np.ones(self.num_action)/self.num_action
        # Transformed utilities
        U_trans = (np.e ** (self.U / self.k)) 
         
        dist = 100      # set initial distance
        k = 0
        
        while dist > tol and k < maxiter:
            # Compute the optimal experiment
            P = ((U_trans * q) /
                 (U_trans * q).sum(axis=1, keepdims=True))
            # Compute the updated unconditional probabilities
            q_new = self.mu @ P
            # Compute the distance
            dist = euclidean(q, q_new)
            q = q_new # update
            k += 1
        
        if k < maxiter and verbose:
            print(f'Converged after {k} iterations.')
        elif k == maxiter:
            print('Exited without convergence')
        
        self.opt_q = q
        self.opt_exp = P
        
        return q, P


# -------------------------------
#    Class attributes starting
# -------------------------------

    np.set_printoptions(precision=8, suppress=True)


    @property
    def unconditional_prob(self):
        """Returns the unconditional choice probabilities."""
        return self.opt_q

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
