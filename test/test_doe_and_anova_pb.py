from doenova import doe_pb
from numpy.random import randn

doe = doe_pb()

plan = doe.make_plan(4,1,1,1)
nb_y = len(plan)

doe.quantify_levels([[100,200],
                     [50,60],
                     [-25,0],
                     [15,16]])

doe.insert_results( randn(nb_y) )

# Factor selection
sel = [[1, 0, 0, 0], # A
       [0, 0, 1, 0]] # C
anova_comps = doe.anova(sel)

doe.show_model()
doe.show_regress_stats()

# Prediction from model          A    B     C     D
ypred= doe.predict_from_model([[150,  55, -12.5, 15.5],
                               [150,  55, -35,   17]])