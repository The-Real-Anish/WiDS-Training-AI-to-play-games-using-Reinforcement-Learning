from bandits import Bandit
import numpy as np
# Import libraries if you need them

class Agent:
    def __init__(self, bandit: Bandit) -> None:
        self.bandit = bandit
        self.banditN = bandit.getN()

        self.rewards = 0
        self.numiters = 0
    

    def action(self) -> int:
        '''This function returns which action is to be taken. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    def update(self, choice : int, reward : int) -> None:
        '''This function updates all member variables you may require. It must be implemented in the subclasses.'''
        raise NotImplementedError()

    # dont edit this function
    def act(self) -> int:
        choice = self.action()
        reward = self.bandit.choose(choice)

        self.rewards += reward
        self.numiters += 1

        self.update(choice,reward)
        return reward

class GreedyAgent(Agent):

    def __init__(self, bandits: Bandit, initialQ) -> None:
        
        super().__init__(bandits)
        assert(len(initialQ) == self.banditN)
        
        #number of times each action is chosen, and initial Q values for all actions, are set
        self.Q = initialQ
        self.action_counter = np.zeros(self.banditN)
        
    def action(self) -> int:

        return np.argmax(self.Q)

    def update(self, action: int, reward: int) -> None:
        assert 0 <= action < self.banditN
        
        self.action_counter[action] += 1
        # - - - - - - - - This part below is our initial_1_k
        self.Q[action] += (1/self.action_counter[action])*(reward - self.Q[action])
        return

class epsGreedyAgent(GreedyAgent):

    def __init__(self, bandits: Bandit, epsilon : float, initialQ) -> None:
        
        self.epsilon = epsilon
        super().__init__(bandits, initialQ)
    
    def action(self) -> int:

        o_action = super().action()
        #We have stochastics!
        prob = np.full(self.banditN, (self.epsilon / (self.banditN - 1)))
        prob[o_action] = 1 - self.epsilon

        return np.argmax(np.random.multinomial(1, prob))

    def update(self, action: int, reward: int) -> None:

        return super().update(action, reward)

class UCBAAgent(GreedyAgent):
    
    def __init__(self, bandits: Bandit, initialQ, c: float) -> None:
        
        self.c = c
        super().__init__(bandits, initialQ)

    def action(self) -> int:
        
        if np.all(self.action_counter):

            return np.argmax(self.Q + self.c*np.sqrt(self.numiters/self.action_counter))
        else:
            #We'll try to try everything, till np.all evaluates to True
            return np.argmin(self.action_counter)

    def update(self, action: int, reward: int) -> None:
        
        return super().update(action, reward)

class GradientBanditAgent(Agent):

    def __init__(self, bandits: Bandit, alpha : float, pref) -> None:
        
        super().__init__(bandits)
        self.alpha = alpha
        self.H = pref
        #this is our average reward over time
        self.R = 0

    def action(self) -> int:
        
        #softmax distribution preference
        return np.argmax( np.random.multinomial(1, np.exp(self.H)/ np.sum(np.exp(self.H)) ) )

    def update(self, action: int, reward: int) -> None:

        policy = np.exp(self.H)/np.sum(np.exp(self.H))

        self.H += self.alpha * (reward - self.R) * (np.eye(1, self.banditN, k=action)[0] - policy)
        self.R = (self.R * (self.numiters - 1) + reward)/self.numiters

class ThompsonSamplerAgent(Agent):

    def __init__(self, bandits: Bandit) -> None:
        super().__init__(bandits)
        
        if self.bandit.type == "Bernoulli":
            self.initial_1 = np.full(self.banditN, 2.3)
            self.initial_2 = np.full(self.banditN, 2.3)

        else:
            self.initial_1 = np.ones(self.banditN)
            self.initial_2 = np.zeros(self.banditN)
 
    def action(self) -> int:
        
        if self.bandit.type == "Bernoulli":
            return np.argmax(np.random.beta(self.initial_1, self.initial_2))

        else:
            return np.argmax(np.random.normal(loc= self.initial_2, scale= self.first))

    # implement
    def update(self, action: int, reward: int) -> None:
        
        if self.bandit.type == "Bernoulli":
            self.initial_1[action] += reward
            self.initial_2[action] += 1 - reward

        else:
            self.initial_1[action] = self.initial_1[action]/np.sqrt(np.square(self.initial_1[action]) + 1)
            self.initial_2[action] = (reward*np.square(self.initial_1[action]) + self.initial_2[action])/(np.square(self.initial_1[action]) + 1)