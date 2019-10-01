from doenova import doe_2f

doe = doe_2f()
plan = doe.make_plan(4,1,3,1,2)
# 4 factors, 1 generator, 3 replicates, 
# 1 center point in each block,
# 2 blocks

doe.show_generators()
doe.export_plan("plan1", True)
# True as 2nd argument forces the randomization of the runs.