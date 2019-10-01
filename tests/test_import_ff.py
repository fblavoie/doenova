from doenova import doe_2f

doe = doe_2f()
doe.import_plan("plan1_with_results")

#       A  B  C  D
sel = [[1, 0, 0, 0], # A
       [0, 1, 1, 0]] # BC
anova_comps = doe.anova(sel)

doe.show_model()
doe.show_regress_stats()

#                                  A    B    C    D
ypred = doe.predict_from_model([-.25, .5, -.75, .75])