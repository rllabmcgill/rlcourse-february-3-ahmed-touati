import numpy as np
import matplotlib.pylab as plt
import time


class MDP(object):
    def __init__(self, d, gamma):
        self.d = d
        self.gamma = gamma
        # reward vector
        self.R = np.random.rand(d)

        # transition matrix
        P = np.random.rand(d, d)
        self.P = P / np.sum(P, axis=1).reshape(-1, 1)

        self.V_optimal = np.dot(np.linalg.inv(np.eye(self.d) - self.gamma*self.P), self.R).reshape([-1, ])

    # policy evaluation iterative model
    def iterative_method(self, epsilon):
        V = np.zeros(self.d)
        erreur = np.max(np.abs(self.V_optimal - V))
        thresold = epsilon * erreur
        erreurs = [erreur]

        """
        while erreur > thresold:
            V = R + gamma * np.dot(P, V)
            erreur = np.linalg.norm(V_optimal - V, ord=np.inf)
            erreurs.append(erreur)
        """
        n = 0
        begin = time.time()
        while erreur > thresold:
            n += 1
            old_V = V.copy()
            for state in np.arange(self.d):
                V[state] = self.R[state] + np.sum(self.P[state] * old_V)
            erreur = np.abs(self.V_optimal[0] - V[0])
            erreurs.append(erreur)
        work = time.time() - begin
        return work, erreurs
    # Monte Carlo

    def sampler(self, state):
        pi = self.P[state]
        new_state = np.random.choice(self.d, p=pi)
        return new_state

    def wasow_method(self, state,length=40):
        Rewards = [self.R[state]]
        new_state = state
        for k in np.arange(1, length):
            new_state = self.sampler(new_state)
            Rewards.append(self.R[new_state] * self.gamma**k)
        v = np.sum(Rewards)
        return v

    def monte_carlo_method(self, epsilon):
        V = np.zeros(self.d)
        erreur = np.max(np.abs(self.V_optimal - V))
        thresold = epsilon * erreur
        erreurs = [erreur]

        n = 0
        begin = time.time()
        state = 0
        v = V[state]
        while erreur > thresold:
            n += 1
            tau = 1./n
            new_v = self.wasow_method(state, length=100)
            v = (1 - tau) * v + tau * new_v
            erreur = np.abs(self.V_optimal[0] - v)
            erreurs.append(erreur)
        work = time.time() - begin

        return work, erreurs

if __name__ =='__main__':
    """
    mdp = MDP(d=500, gamma=0.5)
    work_iterative = list()
    work_monte_carlo = list()
    for eps in np.arange(0.1, 0, -0.0001):
        work, _ = mdp.iterative_method(eps)
        work_iterative.append(work)
        print 'iterative model work:', work
        work, _ = mdp.monte_carlo_method(eps)
        work_monte_carlo.append(work)
        print 'monte carlo work: ', work

    plt.plot(1 / np.arange(0.1, 0, -0.01), work_iterative, label='iterative')
    plt.plot(1 / np.arange(0.1, 0, -0.01), work_monte_carlo, label='Monte Carlo')
    plt.legend()
    plt.savefig('complexity_10_states.png')
    """
    eps = 0.1
    work_iterative = list()
    work_monte_carlo = list()
    state_dim_List = np.arange(10, 1000, 5)
    num_exp_monte_carlo = 500
    num_exp_iterative = 10
    for d in state_dim_List:
        mdp = MDP(d, gamma=0.5)
        work = 0
        for _ in np.arange(num_exp_iterative):
            w, _ = mdp.iterative_method(eps)
            work += w
        work /= num_exp_iterative
        work_iterative.append(work)
        print '---------------- d = %d ---------------' % d
        print 'iterative model work:', work
        work = 0
        for _ in np.arange(num_exp_monte_carlo):
            w, _ = mdp.monte_carlo_method(eps)
            work += w
        work /= num_exp_monte_carlo
        work_monte_carlo.append(work)
        print 'monte carlo work: ', work
    plt.plot(state_dim_List, work_iterative, label='iterative')
    plt.plot(state_dim_List, work_monte_carlo, label='Monte Carlo')
    plt.xlabel('number of states')
    plt.ylabel('run time')
    plt.legend()
    plt.savefig('complexity_1.png')


