import sys
import time
import numpy as np
from Instance import Instance, NewPatient
from Solution import Solution

WEIGHT_GENDER_MIX = 1e6         # H1
WEIGHT_SURGEON_OVERTIME = 1e4   # H3
WEIGHT_ROOM_CAPACITY = 1e5      # H7
# H2, H5 and H6 are automatically given due to the solution representation


class GeneticPatient:
    def __init__(self, instance: Instance, p: NewPatient):
        self.id = p.id
        self.index = int(p.id.lstrip('p'))
        possible_surgery_days = [d for d in range(0, instance.days) if p.surgery_duration <= instance.surgeons[p.surgeon_id].max_surgery_time[d]]
        self.valid_admission_days = np.array([d for d in range(p.surgery_release_day , p.surgery_due_day + 1) if d in possible_surgery_days])

        self.valid_rooms = np.array([int(rid.lstrip('r')) for rid, r in instance.rooms.items() if r not in p.incompatible_room_ids])

    def get_random_admission_day_for(self):
        return np.random.choice(self.valid_admission_days)
    
    def get_random_room(self):
        return np.random.choice(self.valid_rooms)

class GeneticSolution:
    def __init__(self, instance: Instance, admission_days: np.ndarray, room_assignments: np.ndarray):
        self.instance = instance
        self.admission_days = admission_days
        self.room_assignments = room_assignments
        self.fitness = self.calc_fitness()

    def mutate(self):
        # TODO Dont forget to recalc the fitness
        pass

    def calc_fitness(self):
        fitness = 0
        # what is going on here?
        # this looks like an int array of shape (num_patients,) that is true, if the ith patient was
        # booked to the last day
        scheduled = self.admission_days == self.instance.days
        all_admission_days = np.concatenate((self.admission_days, self.instance.occupant_admission_days))
        all_room_assignments = np.concatenate((self.room_assignments, self.instance.occupant_room_assignments))
        # days is a dict of np.arrays with indices, pointing to patient/occupant indices that have their stay
        # during the current day
        days = {d: np.where((all_admission_days <= d) & (d <= all_admission_days + self.instance.lengths_of_stays))[0] for d in range(self.instance.days)}
        # dict where each unique assigned-to room id points to the patient ids, which were assigned to the current room
        rooms = {self.instance.rooms_to_ids[r] : np.where(all_room_assignments == r)[0] for r in np.unique(all_room_assignments)}

        # for each day
        for _, day_patients in days.items():
            # for each room:booked_patient_ids arr
            for rid, room_patients in rooms.items():
                patients_in_room_on_day = np.intersect1d(day_patients, room_patients)
                if len(patients_in_room_on_day) > 1:
                    # H1 No gender mix in any room
                    genders = self.instance.genders[patients_in_room_on_day]
                    if not np.all(genders == genders[0]):
                        fitness += WEIGHT_GENDER_MIX

                    # H7 Room capacity must be respected
                    if len(patients_in_room_on_day) <= self.instance.rooms[rid].capacity:
                        fitness += WEIGHT_ROOM_CAPACITY

                    # S1 Minimize the mix of age groups
                    fitness += self.instance.weights.room_mixed_age * (np.max(self.instance.ages[patients_in_room_on_day]) - np.min(self.instance.ages[patients_in_room_on_day]))
                

        # H3 No overtime for surgeons
        for day, day_patients in days.items():
            for sid, surgeon_patients in self.instance.surgeon_assignments.items():
                patients_of_surgeon_on_day = np.intersect1d(day_patients, surgeon_patients)
                total_surgery_duration = np.sum(self.instance.surgery_durations[patients_of_surgeon_on_day])
                overtime = total_surgery_duration - self.instance.surgeons[sid].max_surgery_time[day]
                if overtime > 0:
                    fitness += WEIGHT_SURGEON_OVERTIME * overtime

        # S7 Minimize the admission delay
        fitness += self.instance.weights.patient_delay * np.sum(np.where(scheduled, self.admission_days - self.instance.release_days, 0))

        # S8 Schedule as many optional patients as possible
        fitness += self.instance.weights.unscheduled_optional * np.sum(scheduled)
        return fitness
    
    def to_solution(self):
        solution = Solution(self.instance)
        for pid, admission_day, room_assignment in zip(self.instance.patients.keys(), self.admission_days, self.room_assignments):
            if admission_day < self.instance.days:  # last day is seen as unscheduled
                solution.patients[pid].admission_day = int(admission_day)
                solution.patients[pid].room = self.instance.rooms_to_ids[room_assignment]
        return solution
    

class GeneticSolver:
    def __init__(self, instance: Instance, random_seed: int = 42):
        self.instance = instance
        # Modify instance
        for _, p in self.instance.patients.items():
            if not p.mandatory:
                p.surgery_due_day = self.instance.days
        # TODO: shouldnt we add these new variables to the Instance class?
        self.instance.occupant_ids = np.array([])
        self.instance.occupant_admission_days = np.array([0 for _, _ in self.instance.occupants.items()])
        self.instance.occupant_room_assignments = np.array([int(o.room_id.lstrip('r'))  for _, o in self.instance.occupants.items()])
        self.instance.release_days = np.array([p.surgery_release_day for _, p in self.instance.patients.items()])
        self.instance.lengths_of_stays = np.concatenate((np.array([p.length_of_stay for _, p in self.instance.patients.items()]), np.array([o.length_of_stay for _, o in self.instance.occupants.items()])))
        self.instance.ages = np.concatenate((np.array([instance.age_groups.index(p.age_group) for _, p in self.instance.patients.items()]), np.array([instance.age_groups.index(o.age_group) for _, o in self.instance.occupants.items()])))
        self.instance.genders = np.concatenate((np.array([p.gender for _, p in self.instance.patients.items()]), np.array([o.gender for _, o in self.instance.occupants.items()])))
        self.instance.surgery_durations = np.array([p.surgery_duration for _, p in self.instance.patients.items()])
        self.instance.surgeon_assignments = {s: np.where([p.surgeon_id == s for _, p in self.instance.patients.items()])[0] for s in self.instance.surgeons.keys()}
        self.instance.rooms_to_ids = {int(rid.lstrip('r')) : rid for rid, _ in instance.rooms.items()}

        self.patients = [GeneticPatient(self.instance, p) for _, p in self.instance.patients.items()]
        np.random.seed(random_seed)

    def generate_solution(self):
        admission_days = np.array([p.get_random_admission_day_for() for p in self.patients])
        room_assignments = np.array([p.get_random_room() for p in self.patients])
        return GeneticSolution(self.instance, admission_days, room_assignments)
    
    def run(self):
        # TODO The actual Genetic Algorithm 
        # Initialize the population
        # self.population = [self.generate_solution() for _ in range(10)]
        return self.generate_solution().to_solution() # Return the best solution here

    def crossover(gs1: GeneticSolution, gs2: GeneticSolution):
        # TODO generate children from parents
        pass

if __name__ == "__main__":
    instance = Instance.from_file(sys.argv[1])
    start_time = time.time()
    solver = GeneticSolver(instance)
    solution = solver.run()
    end_time = time.time()
    print("Elapsed time: ", (end_time-start_time), "s")
    solution.print_table(len(sys.argv) > 2)
    solution.to_file(sys.argv[1].replace(".json", "_sol.json"))
    


        