import sys
import minizinc
from Instance import Instance
from Solution import Solution
import time
import datetime

VERBOSE = True

def solve_instance(i: Instance):
    """Solves the scheduling problem using MiniZinc."""
    
    solution = Solution(i)
    # Create a MiniZinc instance
    # alternate solvers: gecode, chuffed, couenne
    solver = minizinc.Solver.lookup("chuffed")
    model = minizinc.Model('./model3_simple.mzn')

    # Create an instance of the model
    instance = minizinc.Instance(solver, model)

    genders = [p.gender for _, p in i.patients.items()]
    
    # Populate MiniZinc variables from your Python `Instance` object
    if VERBOSE:
        print("Number of patients:", len(list(i.patients.keys())))
        print("Number of rooms:", len(list(i.rooms.keys())))
        print("Number of surgeons:", len(list(i.surgeons.keys())))  
        print("Weight Unscheduled Optional", i.weights.unscheduled_optional)
        print("Weight Patient Delay", i.weights.patient_delay)
        print("Weight Room Mixed Age", i.weights.room_mixed_age)

    instance["WEIGHT_ROOM_MIXED_AGE"] = i.weights.room_mixed_age
    instance["WEIGHT_PATIENT_DELAY"] = i.weights.patient_delay
    instance["WEIGHT_UNSCHEDULED_OPTIONAL"] = i.weights.unscheduled_optional

    instance["PATIENTS"] = list(i.patients.keys())
    instance["NUM_AGE_GROUPS"] = len(i.age_groups)
    instance["GENDERS"] = list(set(genders))
    instance["NUM_DAYS"] = i.days
    instance["SURGEONS"] = list(i.surgeons.keys())
    instance["ROOMS"] = list(i.rooms.keys())
    instance["OCCUPANTS"] = list(i.occupants.keys())

    instance["is_mandatory"] = [p.mandatory for _, p in i.patients.items()]
    instance["surgery_release_day"] = [p.surgery_release_day for _, p in i.patients.items()]
    instance["surgery_due_day"] = [p.surgery_due_day for _, p in i.patients.items()]
    instance["surgeon_id"] = [p.surgeon_id for _, p in i.patients.items()]
    instance["surgery_duration"] = [p.surgery_duration for _, p in i.patients.items()]
    instance["length_of_stay_patient"] = [p.length_of_stay for _, p in i.patients.items()]
    instance["age_group"] = [i.age_groups.index(p.age_group) for _, p in i.patients.items()]
    instance["gender"] = [p.gender for _, p in i.patients.items()]
    instance["max_surgery_time"] = [s.max_surgery_time for _, s in i.surgeons.items()]
    instance["room_capacity"] = [r.capacity for _, r in i.rooms.items()]
    instance["is_patient_compatible_with_room"] = [[rId not in p.incompatible_room_ids for rId in i.rooms.keys()] for _, p in i.patients.items()]
    instance["occupant_length_of_stay"] = [o.length_of_stay for _, o in i.occupants.items()]
    instance["occupant_room_booking"] = [o.room_id for _, o in i.occupants.items()]
    instance["occupant_gender"] = [o.gender for _, o in i.occupants.items()]
    instance["occupant_age_group"] = [i.age_groups.index(o.age_group) for _, o in i.occupants.items()]

    # Solve the problem
    result = instance.solve(
        time_limit=datetime.timedelta(seconds=600),    # 600 seconds
        processes=4                                   # 4 threads (solver processes)
    )

    if result:
        for pId, admission_day, room_assignment in zip(i.patients.keys(), result["patient_admission_day"], result["patient_room_booking"]):
            # print(f"Patient {pId} is {"mandatory" if i.patients[pId].mandatory else "not mandatory"} and has admission on day {admission_day}")
            if admission_day < i.days:
                solution.patients[pId].admission_day = admission_day
                solution.patients[pId].room = room_assignment
                solution.patients[pId].operating_theater = "t0" # Default to avoid error with validator
        
    else:
        print("No Solution found!")
    return solution


if __name__ == '__main__':
    instance = Instance.from_file(sys.argv[1])
    start_time = time.time()
    solution = solve_instance(instance)
    end_time = time.time()
    print("Elapsed time: ", (end_time-start_time), "s")
    solution.print_table(len(sys.argv) > 2)
    solution.to_file(sys.argv[1].replace(".json", "_solution_model3_simple.json"))