import json
import sys
import time
import warnings
from pprint import pprint
import os
import Validator
import numpy as np
from Instance import Instance, NewPatient
from Solution import Solution
import matplotlib.pyplot as plt

# For Objective 1
WEIGHT_GENDER_MIX = 1e6  # H1
WEIGHT_SURGEON_OVERTIME = 1e4  # H3
WEIGHT_ROOM_CAPACITY = 1e5  # H7
# H2, H5 and H6 are automatically given due to the solution representation

# For Objective 2
WEIGHT_HARD_CONSTRAINT = 1e10

class GeneticPatient:
    def __init__(self, instance: Instance, p: NewPatient):
        self.id = p.id
        self.index = int(p.id.lstrip('p'))
        self.mandatory = p.mandatory
        possible_surgery_days = [d for d in range(0, instance.days) if
                                 p.surgery_duration <= instance.surgeons[p.surgeon_id].max_surgery_time[d]]
        self.valid_admission_days = np.array(
            [d for d in range(p.surgery_release_day, p.surgery_due_day + 1) if d in possible_surgery_days or d == instance.days])
        
        self.valid_rooms = np.array(
            [int(rid.lstrip('r')) for rid, r in instance.rooms.items() if r not in p.incompatible_room_ids])
        
        valid_rooms_capacities = np.array([instance.rooms[instance.rooms_to_ids[r]].capacity for r in self.valid_rooms])
        self.valid_rooms_weights = valid_rooms_capacities / sum(valid_rooms_capacities)
        

    def get_random_admission_day(self):
        if self.mandatory or np.random.choice([True, False]):
            return np.random.choice(self.valid_admission_days)
        else:
            return self.valid_admission_days.max()  # unschedule the patient 

    def get_random_room(self):
        return np.random.choice(self.valid_rooms, p=self.valid_rooms_weights)
    


class GeneticSolution:
    def __init__(self, instance: Instance, admission_days: np.ndarray, room_assignments: np.ndarray):
        self.instance = instance
        self.admission_days = admission_days
        self.room_assignments = room_assignments
        self.fitness = self.calc_fitness()  # the higher the better

    def calc_fitness(self):
        fitness = 0
        scheduled = self.admission_days < self.instance.days
        all_admission_days = np.concatenate((self.admission_days, self.instance.occupant_admission_days))
        all_room_assignments = np.concatenate((self.room_assignments, self.instance.occupant_room_assignments))
        # days is a dict of np.arrays with indices, pointing to patient/occupant indices that have their stay
        # during the current day
        days = {d: np.where((all_admission_days <= d) & (d <= all_admission_days + self.instance.lengths_of_stays))[0]
                for d in range(self.instance.days)}
        # dict where each unique assigned-to room id points to the patient ids, which were assigned to the current room
        rooms = {self.instance.rooms_to_ids[r]: np.where(all_room_assignments == r)[0] for r in
                 np.unique(all_room_assignments)}

        # for each day: patients booked on that day
        for _, day_patients in days.items():
            # for each room:booked_patient_ids arr
            for rid, room_patients in rooms.items():
                # get the patient ids that are in the current room only on the current day
                patients_in_room_on_day = np.intersect1d(day_patients, room_patients)
                if len(patients_in_room_on_day) > 1:
                    # H1 No gender mix in any room
                    genders = self.instance.genders[patients_in_room_on_day]
                    # if not all genders are equal
                    if not np.all(genders == genders[0]):
                        fitness += WEIGHT_GENDER_MIX

                    # H7 Room capacity must be respected
                    if len(patients_in_room_on_day) > self.instance.rooms[rid].capacity:
                        fitness += WEIGHT_ROOM_CAPACITY * (
                                    len(patients_in_room_on_day) - self.instance.rooms[rid].capacity)

                    # S1 Minimize the mix of age groups
                    fitness += self.instance.weights.room_mixed_age * (
                                np.max(self.instance.ages[patients_in_room_on_day]) - np.min(
                            self.instance.ages[patients_in_room_on_day]))

        # H3 No overtime for surgeons
        # for each day: patients_ids_booked_to_that_day (arr)
        for day, day_patients in days.items():
            # for each surgeon_id, patient_ids_surgeon_was_assigned_to
            for sid, surgeon_patients in self.instance.surgeon_assignments.items():
                # get all patients that were booked for surgery on this day with this surgeon
                patients_of_surgeon_on_day = np.intersect1d(day_patients, surgeon_patients)
                # get the total surgery duration of patients_of_surgeon_on_day
                total_surgery_duration = np.sum(self.instance.surgery_durations[patients_of_surgeon_on_day])
                # compute any overtime and penalize the fitness accordingly
                overtime = total_surgery_duration - self.instance.surgeons[sid].max_surgery_time[day]
                if overtime > 0:
                    fitness += WEIGHT_SURGEON_OVERTIME * overtime

        # S7 Minimize the admission delay
        # no need to bound it to zero because patients can never be booked early anyway
        fitness += self.instance.weights.patient_delay * np.sum(
            np.where(scheduled, self.admission_days - self.instance.release_days, 0))

        # S8 Schedule as many optional patients as possible
        fitness += self.instance.weights.unscheduled_optional * (len(scheduled) - np.sum(scheduled))
        return -fitness

    def to_solution(self):
        solution = Solution(self.instance)
        for pid, admission_day, room_assignment in zip(self.instance.patients.keys(), self.admission_days,
                                                       self.room_assignments):
            if admission_day < self.instance.days:  # last day is seen as unscheduled
                solution.patients[pid].admission_day = int(admission_day)
                solution.patients[pid].room = self.instance.rooms_to_ids[room_assignment]
        return solution
    

class GeneticSolution2(GeneticSolution):
    def calc_fitness(self):
        fitness = 0
        scheduled = self.admission_days < self.instance.days
        all_admission_days = np.concatenate((self.admission_days, self.instance.occupant_admission_days))
        all_room_assignments = np.concatenate((self.room_assignments, self.instance.occupant_room_assignments))
        patients_per_day = {d: np.where((all_admission_days <= d) & (d <= all_admission_days + self.instance.lengths_of_stays))[0]
                for d in range(self.instance.days)}
        patients_per_room = {self.instance.rooms_to_ids[r]: np.where(all_room_assignments == r)[0] for r in
                 np.unique(all_room_assignments)}

        for _, day_patients in patients_per_day.items():
            for rid, room_patients in patients_per_room.items():
                patients_in_room_on_day = np.intersect1d(day_patients, room_patients)
                if len(patients_in_room_on_day) > 1:
                    # H1 No gender mix in any room
                    genders = self.instance.genders[patients_in_room_on_day]
                    unique, counts = np.unique(genders, return_counts=True)
                    # if not all genders are equal
                    if len(unique) > 1:
                        fitness += (counts.min() ** 2) * WEIGHT_HARD_CONSTRAINT

                    # H7 Room capacity must be respected
                    if len(patients_in_room_on_day) > self.instance.rooms[rid].capacity:
                        fitness += ((len(patients_in_room_on_day) - self.instance.rooms[rid].capacity) ** 2) * WEIGHT_HARD_CONSTRAINT

                    # S1 Minimize the mix of age groups
                    fitness += self.instance.weights.room_mixed_age * (np.max(self.instance.ages[patients_in_room_on_day]) - np.min(self.instance.ages[patients_in_room_on_day]))

        # H3 No overtime for surgeons
        for day, day_patients in patients_per_day.items():
            for sid, surgeon_patients in self.instance.surgeon_assignments.items():
                patients_of_surgeon_on_day = np.intersect1d(day_patients, surgeon_patients)
                total_surgery_duration = np.sum(self.instance.surgery_durations[patients_of_surgeon_on_day])
                overtime = total_surgery_duration - self.instance.surgeons[sid].max_surgery_time[day]
                if overtime > 0:
                    avg_surgery_duration = total_surgery_duration / len(patients_of_surgeon_on_day)
                    fitness += (np.ceil(overtime/avg_surgery_duration) ** 2) * WEIGHT_HARD_CONSTRAINT

        # S7 Minimize the admission delay
        fitness += self.instance.weights.patient_delay * np.sum(np.where(scheduled, self.admission_days - self.instance.release_days, 0))

        # S8 Schedule as many optional patients as possible
        fitness += self.instance.weights.unscheduled_optional * (len(scheduled) - np.sum(scheduled))
        return -fitness
        

class GeneticSolver:
    def __init__(self, instance: Instance, instance_path=None, random_seed: int = 42):
        self.instance = instance
        self.instance_path = instance_path
        # Modify instance
        for _, p in self.instance.patients.items():
            if not p.mandatory:
                p.surgery_due_day = self.instance.days
        self.instance.occupant_ids = np.array([])
        self.instance.occupant_admission_days = np.array([0 for _, _ in self.instance.occupants.items()])
        self.instance.occupant_room_assignments = np.array(
            [int(o.room_id.lstrip('r')) for _, o in self.instance.occupants.items()])
        self.instance.release_days = np.array([p.surgery_release_day for _, p in self.instance.patients.items()])
        self.instance.due_days = np.array([p.surgery_due_day for _, p in self.instance.patients.items()])
        self.instance.lengths_of_stays = np.concatenate((np.array(
            [p.length_of_stay for _, p in self.instance.patients.items()]), np.array(
            [o.length_of_stay for _, o in self.instance.occupants.items()])))
        self.instance.ages = np.concatenate((np.array(
            [instance.age_groups.index(p.age_group) for _, p in self.instance.patients.items()]), np.array(
            [instance.age_groups.index(o.age_group) for _, o in self.instance.occupants.items()])))
        self.instance.genders = np.concatenate((np.array([p.gender for _, p in self.instance.patients.items()]),
                                                np.array([o.gender for _, o in self.instance.occupants.items()])))
        self.instance.surgery_durations = np.array([p.surgery_duration for _, p in self.instance.patients.items()])
        self.instance.surgeon_assignments = {
            s: np.where([p.surgeon_id == s for _, p in self.instance.patients.items()])[0] for s in
            self.instance.surgeons.keys()}
        self.instance.rooms_to_ids = {int(rid.lstrip('r')): rid for rid, _ in instance.rooms.items()}

        self.patients = [GeneticPatient(self.instance, p) for _, p in self.instance.patients.items()]
        np.random.seed(random_seed)

    def generate_solution(self):
        # for each patient, generate a random admission day
        # (these are automatically between release and due date, so H_ is never violated)
        admission_days = np.array([p.get_random_admission_day() for p in self.patients])
        # for each patient, book him to a random valid room
        # (these are automatically compatible, so H_ is never violated)
        room_assignments = np.array([p.get_random_room() for p in self.patients])
        return GeneticSolution2(self.instance, admission_days, room_assignments)

    # selection returns the selection probabilities for each population member
    @staticmethod
    def selection(population: [GeneticSolution], fitnesses: np.ndarray, selection_algorithm):
        return selection_algorithm(population, fitnesses)

    @staticmethod
    def roulette_selection(population: [GeneticSolution], fitnesses: np.ndarray):
        # if all fitnesses are equal, the weights are equiprobable
        if np.min(fitnesses) == np.max(fitnesses):
            n = len(fitnesses)
            weights = np.repeat(1 / n, n)
        # else, inverse the fitnesses so that a higher negative fitness
        # will be weighted more, then normalize the weights
        else:
            weights = (np.min(fitnesses) - fitnesses)
            weights = weights / np.sum(weights)
        return weights

    @staticmethod
    def linear_ranked_selection(population: [GeneticSolution], fitnesses: np.ndarray):
        # np.argsort computes reversed ranks of the fitnesses
        reverse_ranks = np.argsort(np.argsort(fitnesses))
        # print("Reverse ranks:")
        # pprint(reverse_ranks)
        # to use them as weights, normalize them
        weights = reverse_ranks / np.sum(reverse_ranks)
        return weights

    @staticmethod
    def exponential_ranked_selection(population: [GeneticSolution], fitnesses: np.ndarray):
        # np.argsort computes reversed ranks of the fitnesses
        reverse_ranks = np.exp(np.argsort(np.argsort(fitnesses)))
        # to use them as weights, normalize them
        weights = reverse_ranks / np.sum(reverse_ranks)
        return weights

    @staticmethod
    def crossover(mother: GeneticSolution, father: GeneticSolution, crossover_algorithm, weighted=False):
        return crossover_algorithm(mother, father, weighted)

    @staticmethod
    def random_crossover(mother: GeneticSolution, father: GeneticSolution, weighted=False):
        # sample from mother and father solution arrays randomly
        # optionally, the counts of solution values are weighted by the parents' fitness
        if weighted:
            weights = np.array([mother.fitness, father.fitness])
            weights = weights / np.sum(weights)
        else:
            weights = np.array([.5, .5])
        # generate an array of indices to pick from the mother (0) or the father (1)
        day_selector = np.random.choice([0, 1], size=len(mother.admission_days), replace=True, p=weights)
        room_selector = np.random.choice([0, 1], size=len(mother.room_assignments), replace=True, p=weights)
        return GeneticSolver.perform_crossover(mother, father, day_selector, room_selector)
    
    @staticmethod
    def random_crossover_uniform(mother: GeneticSolution, father: GeneticSolution, weighted=False):
        # sample from mother and father solution arrays randomly
        # optionally, the counts of solution values are weighted by the parents' fitness
        if weighted:
            weights = np.array([mother.fitness, father.fitness])
            weights = weights / np.sum(weights)
        else:
            weights = np.array([.5, .5])
        # generate an array of indices to pick from the mother (0) or the father (1)
        selector = np.random.choice([0, 1], size=len(mother.admission_days), replace=True, p=weights)
        return GeneticSolver.perform_crossover(mother, father, selector, selector)
    
    
    @staticmethod
    def single_point_crossover(mother: GeneticSolution, father: GeneticSolution, weighted=False):
        num_days = len(mother.admission_days)
        num_rooms = len(mother.room_assignments)
        # by default, not weighted, the crossover point is at 50%
        if not weighted:
            days_crossover_point = num_days // 2
            rooms_crossover_point = num_rooms // 2
        # else, if weighted, get the weight of the mother's fitness and set the
        # crossover point at that ratio
        else:
            weights = np.array([mother.fitness, father.fitness])
            weights = weights / np.sum(weights)
            days_crossover_point = round(num_days * weights[0])
            rooms_crossover_point = round(num_rooms * weights[0])
        # generate an array of indices to pick from the mother (0) or the father (1)
        day_selector = (np.arange(num_days) > days_crossover_point).astype(int)
        room_selector = (np.arange(num_rooms) > rooms_crossover_point).astype(int)
        return GeneticSolver.perform_crossover(mother, father, day_selector, room_selector)

    @staticmethod
    def double_point_crossover(mother: GeneticSolution, father: GeneticSolution, weighted=False):
        num_days = len(mother.admission_days)
        num_rooms = len(mother.room_assignments)
        # by default, not weighted, the crossover point is at 25% and 75%
        if not weighted:
            days_crossover_point_A = num_days // 4
            days_crossover_point_B = days_crossover_point_A * 3
            rooms_crossover_point_A = num_rooms // 4
            rooms_crossover_point_B = rooms_crossover_point_A * 3
        # else, if weighted, get the weight of the mother's fitness and set the
        # crossover point at that ratio
        else:
            weights = np.array([mother.fitness, father.fitness])
            weights = weights / np.sum(weights)
            # get the indices at the corresponding quantiles
            days_crossover_point_A = round(num_days * weights[1] / 2)
            days_crossover_point_B = round(num_days * (weights[0] + weights[1] / 2))
            rooms_crossover_point_A = round(num_rooms * weights[1] / 2)
            rooms_crossover_point_B = round(num_rooms * (weights[0] + weights[1] / 2))
        # generate an array of indices to pick from the mother (0) or the father (1)
        day_selector = np.logical_and(np.arange(num_days) > days_crossover_point_A,
                                      np.arange(num_days) <= days_crossover_point_B).astype(int)
        room_selector = np.logical_and(np.arange(num_rooms) > rooms_crossover_point_A,
                                       np.arange(num_rooms) <= rooms_crossover_point_B).astype(int)
        return GeneticSolver.perform_crossover(mother, father, day_selector, room_selector)

    @staticmethod
    def perform_crossover(mother: GeneticSolution, father: GeneticSolution, admission_day_indices,
                          room_assignment_indices):
        # each parent attribute contains only generally valid solutions
        # each solutions consists of 2 arrays
        # you can just pick samples of the arrays to generate a new thing
        # generate weights of whom to pick from (optional)
        # admission_day_indices, room_assignment_indices: arrays of zeros and ones
        # pick a solution from the mother for 0 and from the father for 1
        child_admission_days = np.where(admission_day_indices == 1, mother.admission_days, father.admission_days)
        child_room_assignments = np.where(room_assignment_indices == 1, mother.room_assignments,
                                          father.room_assignments)
        return GeneticSolution2(mother.instance, child_admission_days, child_room_assignments)

    def run(self, output_path, selection_algorithm=None, crossover_algorithm=None, population_size: int = 1000, max_generations: int = 100,
            mutation_rate: float = 0.1, random_seed: int = 42, elitism: float = 0.1,
            crossover_weighted=False, improvement_patience=10, title="", plot=True, verbose=True, equal_axes=True
            ):
        # selection_algorithm: selection function (of this class for example):
        # roulette_selection()
        # ...
        # ...
        if selection_algorithm is None:
            selection_algorithm = self.random_crossover
        # crossover_algorithm: crossover function (of this class for example):
        # random_crossover()
        # ...
        # ...
        if crossover_algorithm is None:
            crossover_algorithm = self.random_crossover
        # population_size: number of solutions per generation
        # max_generations: early quitting criterion
        # mutation_rate: probability of attribute being randomly changed
        # min_improvement: if change of best fitness does not improve while still being positve -> early quitting
        # elitism: proportion of "best" solutions that are copied to next population exactly:
        # eg 0.03 with a population of 100 means the top 3 best solutions are copied to the next generation
        # crossover_weighted: whether to favor the better parent when doing crossovers
        # patience: number of generations of no improvement to allow, gets reset upon improvement

        # set seed
        np.random.seed(random_seed)

        # Initialize the population
        population = [self.generate_solution() for _ in range(population_size)]
        best_fitness = -np.inf
        mean_fitness_per_generation = []
        best_fitness_per_generation = []

        patience = improvement_patience

        if verbose: print(f"Starting Genetic Algorithm with {population_size} individuals")
        for t in range(max_generations):
            # if t % (max_generations / 100) == 0:
            if verbose: print(f"\rGeneration {t + 1}/{max_generations}", end="", flush=True)
            # print()
            # print(f"Generation {t + 1}/{max_generations}")

            # get a np.array of each individual's fitness
            fitnesses = np.array([solution.fitness for solution in population])
            # print("Fitnesses:")
            # pprint( np.round(fitnesses, 2))
            # get the best fitness of this generation
            new_best_fitness = np.max(fitnesses)

            # if there is improvement and it is smaller than min_improvement -> terminate
            fitness_improvement = new_best_fitness - best_fitness
            if fitness_improvement > 0:
                patience = improvement_patience
                best_fitness = new_best_fitness
            else:
                # print("No improvement:", best_fitness, ">", new_best_fitness)
                patience -= 1

            # update the lists for visualization
            mean_fitness_per_generation.append(np.mean(fitnesses))
            best_fitness_per_generation.append(best_fitness)

            if patience <= 0:
                if verbose: print()
                if verbose: print(f"We ran out of patience at generation {t}")
                break
            # Selection: who gets to have children?
            # higher fitness -> more likely to crossover with others
            # tournament selection, roulette wheel selection, or rank-based selection
            # get a list of probabilities to crossover across all population members
            selection_probabilities = GeneticSolver.selection(population, fitnesses, selection_algorithm)
            # print("Selection probabilities:")
            # pprint(np.round(selection_probabilities, 2))
            # Creating the next generation
            new_population = []
            # get the locations of each population member sorted by their selection_probabilities
            idx = np.argsort(-selection_probabilities)
            # print("idx:", idx)
            # 1. apply elitism
            # the given ratio of the total population means the number of top solutions that are copied down
            for i in range(int(elitism * population_size)):
                # index - 1 is the last id, i.e. the highest probability
                try:
                    new_population.append(population[idx[i]])
                except IndexError as e:
                    print("Did you set elitism as an integer instead of a ratio?", elitism)
                    raise e
            # 2. apply crossover
            while len(new_population) < population_size:
                # pick two parents
                try:
                    mother, father = np.random.choice(population, size=2, replace=False, p=selection_probabilities)
                except ValueError as e:
                    # at Grid Run 1781, some Value error occurred:
                    # Fewer non-zero entries in p than size warnings.warn(str(e))
                    # no clue what that means
                    warnings.warn(str(e))
                    break
                # generate a new solution, using material of two parents
                child = GeneticSolver.crossover(father, mother, crossover_algorithm, weighted=crossover_weighted)
                self.mutate_solution(child, mutation_rate=mutation_rate)
                new_population.append(child)

            # overwrite the population for the new generation
            population = new_population

        ###### Plot the fitness development
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(t + 1), mean_fitness_per_generation, label="Mean Fitness")
        ax.plot(range(t + 1), best_fitness_per_generation, label="Best Fitness", color="green")
        if equal_axes:
            plt.xlim(0, max_generations)
            plt.ylim(-4e8, -1e8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        # ax.set_title(f"Mean Fitness Over Time\nmutation rate: {mutation_rate}, elitism: {elitism}")
        ax.set_title(f"Mean Fitness Over Time — {title}")
        fig.text(0.5, 0.03, ha="center", fontsize=8,
                 s=f"Mutation rate: {mutation_rate}, Elitism: {elitism}, Population size: {population_size},\nSelection: {selection_algorithm.__name__}, Crossover: {crossover_algorithm.__name__} {'(weighted)' if crossover_weighted else ''}")
        plt.subplots_adjust(bottom=0.2)

        ax.legend()
        if plot and verbose: plt.show()
        plt.savefig(
            f"output/plots/{title.replace('/', '.')}_MR{round(mutation_rate, 4)}_E{round(elitism, 4)}_P{population_size}_{selection_algorithm.__name__}_{crossover_algorithm.__name__}{'_W' if crossover_weighted else ''}.png")
        plt.close("all")

        # celebrate
        if verbose: print()
        if verbose: print(f"Solution found after {t} generations")
        if verbose: print(f"Fitness: {best_fitness}")
        best_solution = population[np.argmax(fitnesses)]
        best_solution_object = best_solution.to_solution()

        # save results to JSON
        # if the reuslts file does ont exist yet, create it
        if output_path is not None:
            if not os.path.exists(output_path):
                with open(output_path, mode='w', encoding='utf-8') as f:
                    json.dump([], f)
            # load the existing results
            with open(output_path, mode='r', encoding='utf-8') as f:
                results = json.load(f)
            # append your new results
            results.append({
                "instance": title,
                "mutation_rate": mutation_rate,
                "elitism": elitism,
                "population_size": population_size,
                "selection_algorithm": selection_algorithm.__name__,
                "crossover_algorithm": crossover_algorithm.__name__,
                "crossover_weighted": crossover_weighted,
                "generations": t,
                "best_fitness": best_fitness,
            })
            # write the updated results back to the json file
            with open(output_path, mode='w', encoding='utf-8') as f:
                json.dump(results, f)

        return best_solution_object  # Return the best solution here

    def mutate_solution(self, solution: GeneticSolution, mutation_rate: float):
        mutated = False
        # num patients is the length of he two arrays that each solution represents
        num_patients = len(solution.admission_days)
        for i in range(num_patients):
            if np.random.random() < mutation_rate:
                mutated = True
                solution.admission_days[i] = self.patients[i].get_random_admission_day()
            if np.random.random() < mutation_rate:
                mutated = True
                solution.room_assignments[i] = self.patients[i].get_random_room()
        # Dont forget to recalc the fitness if there was a mutation
        if mutated:
            solution.fitness = solution.calc_fitness()


def get_values_from_0_to_1(a, r=4):
    return np.append(np.round(np.arange(0, 1, 1 / a), r), 1)


def grid_search(input_folder, output_path, skip=30, equal_axes=True):
    # define parameter ranges
    selection_algos = [GeneticSolver.roulette_selection, GeneticSolver.linear_ranked_selection,
                       GeneticSolver.exponential_ranked_selection]
    crossover_algos = [GeneticSolver.random_crossover]#,
                       #GeneticSolver.single_point_crossover, GeneticSolver.double_point_crossover]
    mutation_rates = np.arange(0.05, 0.5, 0.2)#get_values_from_0_to_1(6)
    elitism_rates = np.arange(.1, 1, 0.25)#get_values_from_0_to_1(6)
    population_sizes = [1000]#[10, 100]
    weighteds = [True, False]
    # for every file in your input folder
    print("Starting grid search")
    i = 0
    for entry in os.scandir(input_folder):
        if entry.is_file():
            instance_file = entry.path
            if "sol" in instance_file or "DS_Store" in instance_file: continue
            print(f"\tFile ({instance_file})")
            instance = Instance.from_file(instance_file)
            solver = GeneticSolver(instance, instance_path=instance_file)
            # run on every possible parameter combination
            # the results will automatically be saved, as well as the learning plots
            # the solutions will not be saved
            # to save a specific solution, run main.py and supply your instance path
            for selection_algo in selection_algos:
                for crossover_algo in crossover_algos:
                    for mutation_rate in mutation_rates:
                        for elitism_rate in elitism_rates:
                            for population_size in population_sizes:
                                for weighted in weighteds:
                                    i += 1
                                    if i < skip: continue
                                    print("Grid Run", i)
                                    solution = solver.run(
                                        output_path=output_path,
                                        selection_algorithm=selection_algo,
                                        crossover_algorithm=crossover_algo,
                                        mutation_rate=mutation_rate,
                                        elitism=elitism_rate,
                                        population_size=population_size,
                                        crossover_weighted = weighted,
                                        title=instance_file,
                                        verbose=False,
                                        plot=False,
                                        equal_axes=equal_axes)

def showcase():
    # run with optimal params
    # execute all these
    for f in os.listdir("ihtc2024_test_dataset"):
        if "sol" in f or "test" not in f: continue
        if "1" in f or "2" in f or "4" in f or "8" in f:
            print()
            print(f)
            for seed in range(5):
                instance_path = os.path.join("ihtc2024_test_dataset", f)
                instance = Instance.from_file(instance_path)
                solver = GeneticSolver(instance, instance_path=instance_path)
                # parameters
                # run the search algorithm
                solution = solver.run(
                    output_path=None,
                    equal_axes=False,
                    mutation_rate=0.05,
                    elitism=0.35,
                    selection_algorithm=GeneticSolver.roulette_selection,
                    population_size=100,
                    crossover_weighted=True,
                    # verbose=False,
                    random_seed=seed)
                solution_path = f"output/final_test/{f}_solution_{seed}.json"
                # solution.print_table()
                solution.to_file(solution_path)
                # validat the solution
                Validator.validate_solution(instance_path, solution_path)


def main(instance_path, output_path=None, **kwargs):
    # perform a single search, save results and visualize the solution
    instance = Instance.from_file(instance_path)
    start_time = time.time()
    solver = GeneticSolver(instance, instance_path=instance_path)
    # parameters
    # run the search algorithm
    solution = solver.run(output_path, equal_axes=False, **kwargs)
    end_time = time.time()
    print("Elapsed time: ", (end_time - start_time), "s")
    solution.print_table(len(sys.argv) > 2)
    solution.to_file(sys.argv[1].replace(".json", "_sol.json"))


if __name__ == "__main__":
    # instance_path = "ihtc2024_test_dataset/test01.json"
    # output_path = "output/genetic_results_experiment2.json"
    # main(output_path=output_path)
    # grid_search("ihtc2024_test_dataset", output_path=output_path, skip=0)
    showcase()



