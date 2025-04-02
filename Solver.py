import sys
import minizinc
from Instance import Instance
from Solution import Solution

def solve_instance(instance: Instance):
    """Solves the scheduling problem using MiniZinc."""

    # Create a MiniZinc instance
    solver = minizinc.Solver.lookup("gecode")
    model = minizinc.Model()
    model.add_string('model.mzn')

    # Create an instance of the model
    instance_data = minizinc.Instance(solver, model)
    
    # Populate MiniZinc variables from your Python `Instance` object
    #instance_data["num_days"] = instance.days
    #instance_data["num_patients"] = len(instance.patients)
    #instance_data["num_rooms"] = len(instance.rooms)
    #instance_data["admission_days"] = [p.surgery_release_day for p in instance.patients.values()]
    #instance_data["length_of_stay"] = [p.length_of_stay for p in instance.patients.values()]

    # Solve the problem
    result = instance_data.solve()
    
    ##
    ## TODO: Add code to extract result data and convert it to an instance
    ## 

    return Solution(instance)
    

if __name__ == '__main__':
    instance = Instance.from_file(sys.argv[1])
    solution = solve_instance(instance)
    solution.print_table(len(sys.argv) > 3)