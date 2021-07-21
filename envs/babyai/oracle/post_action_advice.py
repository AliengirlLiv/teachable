import numpy as np
from envs.babyai.oracle.teacher import Teacher



class PostActionAdvice(Teacher):

    def empty_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([-1])

    def random_feedback(self):
        """
        Return a tensor corresponding to no feedback.
        """
        return np.array([self.env.action_space.sample()])

    def compute_feedback(self):
        """
        Return the expert action from the previous timestep.
        """
        return np.array([self.last_action])

    def feedback_condition(self):
        """
        Returns true when we should give feedback.
        Currently returns true when the agent's past action did not match the oracle's action.
        """
        # For now, we're being lazy and correcting the agent any time it strays from the agent's optimal set of actions.
        # This is kind of sketchy since multiple paths can be optimal.
        return True



