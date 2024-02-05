
import copy
import random as rnd
from functools import reduce


class Evo:

    def __init__(self):
        self.pop = {}       # eval -> solution   eval = ((name1, val1), (name2, val2)..)
        self.fitness = {}   # name -> function
        self.agents = {}    # name -> (operator, num_solutions_input)

    def add_fitness_criteria(self, name, f):
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        popvals = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p, q):
        """ Return whether p dominates q """
        pscores = [score for _,score in p]
        qscores = [score for _,score in q]
        score_diffs = list(map(lambda x,y: y-x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)
        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        return S - {q for q in S if Evo._dominates(p,q)}

    def remove_dominated(self):
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}



    def evolve(self, n=1, dom=100):

        agent_names = list(self.agents.keys())

        for i in range(n):
            # pick an agent
            pick = rnd.choice(agent_names)

            # run the agent to produce a new solution
            self.run_agent(pick)

            # periodically cull the population
            # discard dominated solutions
            if i % dom == 0:
                self.remove_dominated()

        self.remove_dominated()



    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt



