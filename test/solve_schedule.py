import minizinc

class Teacher:
    def __init__(self, name, id):
        self.name = name
        self.id = id

class Class:
    def __init__(self, name, teacher):
        self.name = name
        self.teacher = teacher

# create data
teachers = [
    Teacher("Alice", 1),
    Teacher("Bob", 2),
]
classes = [
    Class("Mathe", teachers[0]),
    Class("Deutsch", teachers[1]),
    Class("Sport", teachers[2]),
]

# create a minizinc model
model = minizinc.Model()
# select the problem
model.add_file("schedule_model.mzn")
# use the default solver
gecode = minizinc.Solver.lookup("gecode")
# crate a model instance
inst = minizinc.Instance(gecode, model)

# assign parameters
inst["NUM_CLASSES"] = len(classes)
inst["NUM_TIMES"] = 3
inst["NUM_ROOMS"] = 2
inst["NUM_TEACHERS"] = len(teachers)
inst["teacher_of_class"] = [c.teacher.id for c in classes]



# solve it
result = inst.solve()

print(result)

