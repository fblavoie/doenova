from doenova import doe_ff
from numpy.random import randn

doe = doe_ff()

plan = doe.make_plan([2,3,2,4,3],3,1)
nb_y = len(plan)

doe.insert_results( randn(nb_y) )

# Factor selection
sel = [[1, 0, 0, 0, 0], # A
       [0, 1, 1, 0, 0], # BC
       [1, 1, 0, 0, 1]] # ABE
anova_comps = doe.anova(sel)
