import minizinc

# create a minizinc model
model = minizinc.Model()
# select the problem
model.add_file("test/send_more_money.mzn")

# use the default solver
gecode = minizinc.Solver.lookup("gecode")

# crate a model instance
inst = minizinc.Instance(gecode, model)
print(inst)

# solve it
result = inst.solve()

print(result)
print(result["digits"])

