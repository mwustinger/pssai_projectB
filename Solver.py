import sys
import minizinc
from Instance import Instance
from Solution import Solution

def solve_instance(instance: Instance):
    """Solves the scheduling problem using MiniZinc."""

    print(instance)

    # Create a MiniZinc instance
    solver = minizinc.Solver.lookup("gecode")
    model = minizinc.Model()
    model.add_string('model.mzn')

    # Create an instance of the model
    instance_data = minizinc.Instance(solver, model)

    genders = [p.gender for _, p in instance.patients.items()]
    
    # Populate MiniZinc variables from your Python `Instance` object
    instance_data["PATIENTS"] = instance.patients.keys()
    instance_data["AGE_GROUPS"] = instance.age_groups
    instance_data["GENDERS"] = list(set(genders))
    instance_data["NUM_DAYS"] = instance.days
    instance_data["SURGEONS"] = instance.surgeons.keys()
    instance_data["ROOMS"] = instance.rooms.keys()

    instance_data["is_mandatory"] = [p.mandatory for _, p in instance.patients.items()]
    instance_data["surgery_release_day"] = [p.surgery_release_day for _, p in instance.patients.items()]
    instance_data["surgery_due_day"] = [p.surgery_due_day for _, p in instance.patients.items()]
    instance_data["surgeon_id"] = [p.surgeon_id for _, p in instance.patients.items()]
    instance_data["surgery_duration"] = [p.surgery_duration for _, p in instance.patients.items()]
    instance_data["length_of_stay"] = [p.length_of_stay for _, p in instance.patients.items()]
    instance_data["age_group"] = [p.age_group for _, p in instance.patients.items()]
    instance_data["gender"] = [p.gender for _, p in instance.patients.items()]
    instance_data["max_surgery_time"] = [s.max_surgery_time for _, s in instance.surgeons.items()]
    instance_data["room_capacity"] = [r.capacity for _, r in instance.rooms.items()]
    instance_data["is_patient_compatible_with_room"] = [[rId not in p.incompatible_room_ids for rId in instance.rooms.keys()] for _, p in instance.patients.items()]
    

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