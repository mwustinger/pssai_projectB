import minizinc
import os
# create a minizinc model
model = minizinc.Model()
# select the problem
# print(os.path)
model.add_file("three_digits.mzn")

# use the default solver
gecode = minizinc.Solver.lookup("gecode")

# crate a model instance
inst = minizinc.Instance(gecode, model)
# set allowed digits manually
digits = list(range(3, 12
                    ))
inst["N"] = len(digits)
inst["allowed"] = digits

# solve it
result = inst.solve()
# result = inst.solve(all_solutions=True)

# for solution in result:
#     print(solution)
print(result)

