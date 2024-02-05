import copy
import random as rnd
from functools import reduce
import pandas as pd
import numpy as np
import time
from profilerlib import Profiler


class Evo:
    """
        Evolutionary computing algorithm for optimizing TA assignments

        Attributes:
        - pop (dict): evaluation to solution mapping
        - fitness (dict): name to fitness function mapping
        - agents (dict): name to agent mapping
        - tas_data (DataFrame): TA data
        - sections_data (DataFrame): sections data
        - upper_limits (dict): user-defined upper limits for each objective
    """

    def __init__(self, tas_file, sections_file, upper_limits):
        """
            Initialize Evo class

            Parameters:
            - tas_file (str): path to TA data file
            - sections_file (str): path to sections data file
            - upper_limits (dict): user-defined upper limits for each objective
        """

        self.pop = {}       # eval -> solution   eval = ((name1, val1), (name2, val2)..)
        self.fitness = {}   # name -> function
        self.agents = {}    # name -> (operator, num_solutions_input)
        self.tas_data, self.sections_data = self.read_data(tas_file, sections_file)
        self.upper_limits = upper_limits


    def read_data(self, tas_file, sections_file):
        """
            Read TA and section data from CSV files

            Parameters:
            - tas_file (str): path to TA data file
            - sections_file (str): path to sections data file

            Returns:
            - tuple: two DataFrames representing TA and section data
        """

        # read in TA and section data from CSVs
        tas_data = pd.read_csv(tas_file)
        sections_data = pd.read_csv(sections_file)

        return tas_data, sections_data


    # OBJECTIVES
    def calculate_overallocation_penalty(self, sol_data):
        """
            Calculate overallocation penalty for a given solution

            Parameters:
            - sol_data (np.array): array representing the assignment of TAs to sections

            Returns:
            - list: list of overallocation penalties for each TA
        """

        # initialize overallocation list
        overallocation = []

        # get maximum sections willing to teach
        max_willing = self.tas_data['max_assigned']

        # sum total amt of assigned sections for each TA
        allocations = sol_data.sum(axis=1)
        for i in range(len(allocations)):
            # if total amt of assigned sections > max willing, calculate overallocation penalty
            # (penalize only if overallocated)
            if allocations[i] > max_willing[i]:
                overallocation.append(allocations[i]-max_willing[i])

        return overallocation



    @Profiler.profile
    def overallocation_objective(self, sol_data_list):
        """
            Calculate the overallocation objective value for a solution

            Parameters:
            - sol_data_list (np.array): arrau representing solution

            Returns:
            - int: total overallocation penalty across all solutions
        """

        penalties = self.calculate_overallocation_penalty(sol_data_list)
        return sum(penalties)


    def has_time_conflicts(self, sol_data):
        """
            Check if a given solution has time conflicts for TAs

            Parameters:
            - sol_data (np.array): arrau representing the assignment of TAs to sections

            Returns:
            - int: number of TAs with time conflicts
        """

        # intialize conficts
        conflicts = 0

        # iterate through each row (TA) in the solution array.
        for i in range(len(sol_data)):

            # get section times of sections the TA is assigned to
            ta_section_times = []
            for section_index in np.where(sol_data[i] == 1)[0]:
                ta_section_times.append(self.sections_data.loc[section_index, 'daytime'])

            # identify time conflicts
            set_times = set(ta_section_times)
            if len(ta_section_times) > len(set_times):
                conflicts += 1

        return conflicts


    @Profiler.profile
    def conflicts_objective(self, sol_data_list):
        """
            Calculate conflicts objective value

            Parameters:
            - sol_data_list (np.array): array of solution data

            Returns:
            - int: conflicts objective value
        """

        penalties = self.has_time_conflicts(sol_data_list)
        return penalties


    def calculate_undersupport_penalty(self, sol_data):
        """
            Calculate undersupport penalty for given solution

            Parameters:
            - sol_data (np.array): array representing the assignment of TAs to sections

            Returns:
            - list: list of undersupport penalties for each section
        """

        # initialize undersupport list
        undersupport = []

        # get minimum TAs needed for each section
        section_min_ta = self.sections_data['min_ta']

        # sum section TA assignments
        column_sums = np.sum(sol_data, axis=0)

        # calculate undersupport (if any)
        for i in range(len(column_sums)):
            support = section_min_ta[i] - column_sums[i]
            if np.any(support < 0):
                support = 0
                undersupport.append(support)
            else:
                undersupport.append(support)

        return undersupport


    @Profiler.profile
    def undersupport_objective(self, sol_data_list):
        """
            Calculate undersupport objective value

            Parameters:
            - sol_data_list (np.array): array of solution data

            Returns:
            - int: undersupport objective value
        """

        penalties = self.calculate_undersupport_penalty(sol_data_list)
        return sum(penalties)


    def calculate_unwilling_penalty(self, sol_data):
        """
            Calculate unwilling penalty for given solution

            Parameters:
            - sol_data (np.array): array representing the assignment of TAs to sections

            Returns:
            - list: list of unwilling penalties for each section
        """

        # get willingness data
        assignment_willingness = self.tas_data.iloc[:, 3:].to_numpy()

        # get 'unwilling' assignments
        indices = np.where((assignment_willingness == 'U') & (sol_data == 1))[0]
        unwilling = len(indices)

        return unwilling


    @Profiler.profile
    def unwilling_objective(self, sol_data_list):
        """
            Calculate unwilling objective value

            Parameters:
            - sol_data_list (np.array): array of solution data

            Returns:
            - int: unwilling objective value
        """

        penalties = self.calculate_unwilling_penalty(sol_data_list)
        return penalties


    def calculate_unpreferred_penalty(self, sol_data):
        """
            Calculate unpreferred penalty for given solution

            Parameters:
            - sol_data (np.array): array representing the assignment of TAs to sections

            Returns:
            - list: list of unpreferred penalties for each section
        """

        # get willingness data
        assignment_willingness = self.tas_data.iloc[:, 3:].to_numpy()

        # get 'willing' assignments
        indices = np.where((assignment_willingness == 'W') & (sol_data == 1))[0]
        unpreferred = len(indices)
        return unpreferred


    @Profiler.profile
    def unpreferred_objective(self, sol_data_list):
        """
            Calculate unpreferred objective value

            Parameters:
            - sol_data_list (np.array): array of solution data

            Returns:
            - int: unpreferred objective value
        """

        penalties = self.calculate_unpreferred_penalty(sol_data_list)
        return penalties


    # AGENTS
    @Profiler.profile
    def random_agent(self, solutions):
        """
            Randomly modify solutions

            Parameters:
            - solutions (np.array): array of solutions to modify

            Returns:
            - np.array: modified solutions
        """

        # rows and cols
        num_solutions, num_tas = solutions.shape

        # get random assignment
        rand_solution = rnd.randint(0, num_solutions - 1)
        rand_ta = rnd.randint(0, num_tas - 1)

        # unassign TA from all sections and randomly assign
        solutions[rand_solution, :] = 0
        solutions[rand_solution, rand_ta] = 1
        return solutions


    @Profiler.profile
    def undersupport_agent(self, solutions):
        """
            Modify solutions to minimize undersupport

            Parameters:
            - solutions (np.array): array of solutions to modify

            Returns:
            - np.array: modified solutions
        """

        # get section w/ most undersupport
        undersupport_penalty = self.calculate_undersupport_penalty(solutions)
        max_undersupport_index = np.argmax(undersupport_penalty)

        # get random TA
        rand_solution = rnd.randint(0, len(solutions) - 1)
        rand_ta = max_undersupport_index

        # unassign TA from all sections and assign to high undersupport
        solutions[rand_solution, :] = 0
        solutions[rand_solution, rand_ta] = 1

        return solutions


    @Profiler.profile
    def overallocation_agent(self, solutions):
        """
            Modify solutions to minimize overallocation

            Parameters:
            - solutions (np.array): array of solutions to modify

            Returns:
            - np.array: modified solutions
        """

        # get overallocation penalty (if any)
        overallocation_penalty = self.calculate_overallocation_penalty(solutions)
        if overallocation_penalty:

            # get most overallocation
            max_overallocation_index = np.argmax(overallocation_penalty)

            # generate random solution for TA w/ most overallocation
            rand_solution = rnd.randint(0, 16)
            rand_ta = max_overallocation_index

            # unassign TA from all sections and assign to high overallocation
            solutions[rand_solution, :] = 0
            solutions[rand_ta, rand_solution] = 1

        return(solutions)


    @Profiler.profile
    def conflict_agent(self, solutions):
        """
            Modify solutions to minimize time conflicts

            Parameters:
            - solutions (np.array): array of solutions to modify

            Returns:
            - np.array: modified solutions
        """

        # get most conflicts
        current_conflicts = self.has_time_conflicts(solutions)
        max_conflicts_index = np.argmax(current_conflicts)

        # generate random solution for TA w/ most conflicts
        rand_solution = rnd.randint(0, len(solutions) - 1)
        rand_ta = max_conflicts_index

        # unassign TA from all sections and assign to section w/ high conflicts
        solutions[rand_solution, :] = 0
        solutions[rand_solution, rand_ta] = 1

        return solutions


    @Profiler.profile
    def preference_agent(self, solutions):
        """
            Modify solutions to prioritize preferred sections

            Parameters:
            - solutions (np.array): array of solutions to modify

            Returns:
            - np.array: modified solutions
        """

        # get willingness data
        assignment_willingness = self.tas_data.iloc[:, 3:].to_numpy()

        # identify preferred sections
        preferred_indices = np.where((assignment_willingness == 'P') & (solutions == 0))

        # if preferred sections available for assignment
        if len(preferred_indices[0]) > 0:

            # randomly choose preferred section, get TA
            rand_index = rnd.choice(range(len(preferred_indices[0])))
            rand_solution, rand_ta = preferred_indices[0][rand_index], preferred_indices[1][rand_index]

            # unassign TA from all sections and assign to preferred section
            solutions[rand_solution, :] = 0
            solutions[rand_solution, rand_ta] = 1

        return solutions


    def add_fitness_criteria(self, name, f):
        """
            Add fitness criteria to the evolutionary algorithm

            Parameters:
            - name (str): name of fitness criteria
            - f (function): fitness function to be added
        """

        self.fitness[name] = f


    def add_agent(self, name, op, k=1):
        """
            Add agent to the evolutionary algorithm

            Parameters:
            - name (str): name of agent
            - op (function): operator function used by agent
            - k (int): n solutions used as input for agent (default is 1)
        """

        self.agents[name] = (op, k)


    def add_solution(self, sol):
        """
            Add solution to the population

            Parameters:
            - sol (np.array): solution to be added to the population
        """

        eval = {'groupname': 'group', **{name: f(sol) for name, f in self.fitness.items()}}
        self.pop[tuple(eval.items())] = sol


    def get_random_solutions(self, k=1):
        """
            Get random solutions from the population

            Parameters:
            - k (int): n random solutions to retrieve (default is 1)

            Returns:
            - np.array: array of random solutions
        """

        popvals = tuple(self.pop.values())
        return copy.deepcopy(rnd.choice(popvals))


    def run_agent(self, name):
        """
            Run specified agent to produce a new solution and add it to the population

            Parameters:
            - name (str): name of  agent to run
        """

        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)


    @staticmethod
    def _dominates(p, q):
        """
            Returns whether solution p dominates solution q

            Parameters:
            - p: solution A
            - q: solution B

            Returns:
            - bool: true if p dominates q, false otherwise
        """

        pscores = [score for _, score in p[1:]]
        qscores = [score for _, score in q[1:]]

        score_diffs = list(map(lambda x, y: y - x, pscores, qscores))
        min_diff = min(score_diffs)
        max_diff = max(score_diffs)

        return min_diff >= 0.0 and max_diff > 0.0


    @staticmethod
    def _reduce_nds(S, p):
        """
            Helper function for removing dominated solutions

            Parameters:
            - S: set of solutions
            - p: solution

            Returns:
            - reduced set of solutions
        """

        return S - {q for q in S if Evo._dominates(p, q)}


    def remove_dominated(self):
        """
            Remove dominated solutions from population
        """

        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k: self.pop[k] for k in nds}


    def remove_violating_solutions(self):
        """
            Remove solutions that violate user-defined upper limits
        """

        to_remove = []

        for eval, sol in self.pop.items():
            # check each objective against upper limit
            for objective, limit in self.upper_limits.items():
                scores = [t[1] for t in eval[1:]]
                for score in scores:
                    if score > limit:
                        to_remove.append(eval)
                        # dont need to check other objectives
                        break

        # remove violating solutions from the population
        for eval in to_remove:
            self.pop.pop(eval, None)


    @Profiler.profile
    def evolve(self, n=1, dom=100, viol=1000, status=1000, time_limit=None, reset=False):
        """

        Run n random agents (default=1)

        dom defines how often we remove dominated (unfit) solutions

        status defines how often we display the current population

        n = # of agent invocations

        dom = interval for removing dominated solutions

        viol = interval for removing solutions that violate user-defined upper limits

        status = interval for display the current population

        time_limit = the evolution time limit (seconds).  Evolve function stops when limit reached

        """

        # get list of agent names
        agent_names = list(self.agents.keys())

        # initialize start time of evolution
        start_time = time.time()

        for i in range(n):

            # pick an agent
            pick = rnd.choice(agent_names)

            # run the agent to produce a new solution
            self.run_agent(pick)

            # periodically cull the population and discard dominated solutions
            if i % dom == 0:
                self.remove_dominated()

            # remove solutions that violate user-defined upper limits
            if i % viol == 0:
                self.remove_violating_solutions()

            # display current population
            if i % status == 0:
                print(f"Current Population at Iteration {i}:\n{self.pop.keys()}")

            # reset population
            if reset and i % reset == 0:
                self.pop = {}
                print("Population reset.")

            # if time limit is specified and exceeded, stop evolving
            if time_limit and (time.time()-start_time) > time_limit:
                break

        self.remove_dominated()


    def get_best_non_dominated_solutions(self):
        """
            Retrieve best non-dominated solutions from the population

            Returns:
            - list: list of non-dominated solution objectives
        """

        nds = [sol for sol in self.pop.keys() if
               not any(self._dominates(sol, other) for other in self.pop.keys() if other != sol)]

        return nds

    def get_best_non_dominated_full_solutions(self):
        """
            Retrieve best non-dominated full solutions from the population

            Returns:
            - list: list of full non-dominated solutions
        """

        nds = [sol for sol in self.pop.keys() if
               not any(self._dominates(sol, other) for other in self.pop.keys() if other != sol)]

        # need solution data to analyze best solution assignments
        full_solutions = []

        for item in nds:
            solution = self.pop[item]
            full_solutions.append({item: solution})

        return full_solutions


    def format_results(self, solutions):
        """
            Format solutions for output in specificied format

            Parameters:
            - solutions (list): list of solutions to be formatted

            Returns:
            - str: formatted string containing  solutions in specified format
        """

        # header for desired format
        formatted_results = ["groupname,overallocation,conflicts,undersupport,unwilling,unpreferred"]

        for sol in solutions:
            # add solutions to formatted results
            eval_dict = dict(sol)
            formatted_results.append(
                f"{eval_dict['groupname']},{eval_dict['Overallocation']},{eval_dict['Conflicts']},{eval_dict['Undersupport']},{eval_dict['Unwilling']},{eval_dict['Unpreferred']}"
            )

        return "\n".join(formatted_results)


    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"
        return rslt







