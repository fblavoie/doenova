from doenova import doe_2f
from numpy.random import randn

doe = doe_2f()

plan = doe.make_plan(4,1,3,1,2)
nb_y = len(plan)

doe.insert_results( randn(nb_y) )

# Factor selection
sel = [[1, 0, 0, 0], # A
       [0, 1, 1, 0]] # BC
anova_comps = doe.anova(sel)

doe.show_model()
doe.show_regress_stats()

# Prediction from model           A    B    C    D
ypred= doe.predict_from_model([[-.25, .5, -.75, .75],
                               [ .5 ,-.9,  .1,  1.3]])