import sys
import minizinc
from Instance import Instance
from Solution import Solution
from pprint import pprint

VERBOSE = True
LIMIT = 0
LIMIT = 130

def solve_instance(instance: Instance, threads = 8):
    """Solves the scheduling problem using MiniZinc."""
    
    solution = Solution(instance)
    # Create a MiniZinc instance
    # alternate solvers: gecode, chuffed, couenne
    solver = minizinc.Solver.lookup("chuffed")
    model = minizinc.Model('./model.mzn')

    # Create an instance of the model
    instance_data = minizinc.Instance(solver, model)
    if LIMIT: 
        instance.patients = dict(list(instance.patients.items())[:LIMIT])

    genders = [p.gender for _, p in instance.patients.items()]
    
    # Populate MiniZinc variables from your Python `Instance` object
    if VERBOSE:
        print("Number of patiens:", len(list(instance.patients.keys())))
        print("Number of rooms:", len(list(instance.rooms.keys())))
        print("Number of surgeons:", len(list(instance.surgeons.keys())))

    instance_data["WEIGHT_ROOM_MIXED_AGE"] = instance.weights.room_mixed_age
    instance_data["WEIGHT_PATIENT_DELAY"] = instance.weights.patient_delay
    instance_data["WEIGHT_UNSCHEDULED_OPTIONAL"] = instance.weights.unscheduled_optional

    instance_data["PATIENTS"] = list(instance.patients.keys())
    instance_data["AGE_GROUPS"] = range(len(instance.age_groups))
    instance_data["GENDERS"] = list(set(genders))
    instance_data["DAYS"] = range(instance.days)
    instance_data["SURGEONS"] = list(instance.surgeons.keys())
    instance_data["ROOMS"] = list(instance.rooms.keys())
    # TODO: es gibt zusätzlich zu Patients auch Occupants, die sind eine Subklassse
    # von Patient und besetzen ein Zimmer ohne gebucht werden zu müssen 
    # und die brauchen auch keine Surgery, die nehmen echt nur Platz ein
    instance_data["OCCUPANTS"] = list(instance.occupants.keys())

    instance_data["is_mandatory"] = [p.mandatory for _, p in instance.patients.items()]
    instance_data["surgery_release_day"] = [p.surgery_release_day for _, p in instance.patients.items()]
    instance_data["surgery_due_day"] = [p.surgery_due_day for _, p in instance.patients.items()]
    instance_data["surgeon_id"] = [p.surgeon_id for _, p in instance.patients.items()]
    instance_data["surgery_duration"] = [p.surgery_duration for _, p in instance.patients.items()]
    instance_data["length_of_stay_patient"] = [p.length_of_stay for _, p in instance.patients.items()]
    instance_data["age_group"] = [instance.age_groups.index(p.age_group) for _, p in instance.patients.items()]
    instance_data["gender"] = [p.gender for _, p in instance.patients.items()]
    instance_data["max_surgery_time"] = [s.max_surgery_time for _, s in instance.surgeons.items()]
    instance_data["room_capacity"] = [r.capacity for _, r in instance.rooms.items()]
    instance_data["is_patient_compatible_with_room"] = [[rId not in p.incompatible_room_ids for rId in instance.rooms.keys()] for _, p in instance.patients.items()]
    instance_data["length_of_stay_occupant"] = [o.length_of_stay for _, o in instance.occupants.items()]
    instance_data["occupant_room_booking"] = [o.room_id for _, o in instance.occupants.items()]
    instance_data["gender_occupant"] = [o.gender for _, o in instance.occupants.items()]
    
    # Solve the problem
    result = instance_data.solve()

    if result:
        for pId, is_scheduled, admission_day, room_assignment in zip(instance.patients.keys(), result["is_scheduled"], result["patient_admission_day"], result["patient_room_booking"]):
            if is_scheduled:
                solution.patients[pId].admission_day = admission_day
                solution.patients[pId].room = room_assignment
    else:
        print("No Solution found!")

    return solution
    

if __name__ == '__main__':
    instance = Instance.from_file(sys.argv[1])
    solution = solve_instance(instance)
    solution.print_table(len(sys.argv) > 2)