from ta_evolution import Evo
import numpy as np
from profilerlib import Profiler

# define upper limits for violations
upper_limits = {
    'Overallocation': 100,
    'Conflicts': 100,
    'Undersupport': 100.0,
    'Unwilling': 100,
    'Unpreferred': 100
}

# create instance of Evo class
evo = Evo('tas.csv', 'sections.csv', upper_limits)

# add objectives
evo.add_fitness_criteria("Overallocation", evo.overallocation_objective)
evo.add_fitness_criteria("Conflicts", evo.conflicts_objective)
evo.add_fitness_criteria("Undersupport", evo.undersupport_objective)
evo.add_fitness_criteria("Unwilling", evo.unwilling_objective)
evo.add_fitness_criteria("Unpreferred", evo.unpreferred_objective)

# add agents
evo.add_agent("RandomAgent", evo.random_agent)
evo.add_agent("UndersupportAgent", evo.undersupport_agent)
evo.add_agent("OverallocationAgent", evo.overallocation_agent)
evo.add_agent("ConflictAwareAgent", evo.conflict_agent)
evo.add_agent("PreferenceAgent", evo.preference_agent)

# intialize population
initial_population_zeros = np.zeros((43, 17))
initial_population_ones = np.ones((43, 17))

evo.add_solution(initial_population_ones)
evo.add_solution(initial_population_zeros)

# run the optimizer
evo.evolve(10**9, dom=100, status=10000, time_limit=600)

# retrieve best non-dominated solutions
best_solutions = evo.get_best_non_dominated_solutions()
best_full_solutions = evo.get_best_non_dominated_full_solutions()

# format solution results
formatted_results = evo.format_results(best_solutions)

# display best solutions, solution results, and Profiler report
print('FULL BEST SOLUTIONS')
print(best_full_solutions)
print('')
print('FORMATTED RESULTS')
print(formatted_results)
print('')
print('PROFILER REPORT')
print(Profiler.report())
