% weights
int: WEIGHT_ROOM_MIXED_AGE;
int: WEIGHT_PATIENT_DELAY;
int: WEIGHT_UNSCHEDULED_OPTIONAL;

% define the sets of possible values of the constants
enum PATIENTS;
set of int: AGE_GROUPS;
enum GENDERS;
set of int: DAYS;
set of int: MINUTES = 0..1339;
enum SURGEONS;
enum ROOMS;
enum OCCUPANTS;

% define the arrays indexed by the sets
array[PATIENTS] of bool: is_mandatory; % a true/false array, where the ith value belongs to the ith patient
array[PATIENTS] of DAYS: surgery_release_day; % from this day surgery can be done
array[PATIENTS] of opt DAYS: surgery_due_day; % until this day surgery must be finished
array[PATIENTS] of SURGEONS: surgeon_id; % the surgeon who carries out the patient’s surgery
array[PATIENTS] of MINUTES: surgery_duration; % the expected duration of the patient’s surgery, which is assumed to always take place on the day of admission.
array[PATIENTS] of DAYS: length_of_stay_patient; % duration of the hospitalization in days.
array[PATIENTS] of AGE_GROUPS: age_group; % the age group of the patient (e.g., infant, youth, adult, elder). The list of age groups is fully ordered.
array[PATIENTS] of GENDERS: gender; % the gender of the patient.
array[SURGEONS, DAYS] of MINUTES: max_surgery_time; % max duration a surgeon can work each day
array[ROOMS] of int: room_capacity; % max number of Patients allowed in here
array[PATIENTS, ROOMS] of bool: is_patient_compatible_with_room; % 2D matrix with patients as rows and rooms as cols
array[OCCUPANTS] of DAYS: occupant_length_of_stay; % duration of the occupant's hospitalization in days.
array[OCCUPANTS] of ROOMS: occupant_room_booking; % room that the occupant is already inside of
array[OCCUPANTS] of GENDERS: occupant_gender; % the gender of the occupant.
array[OCCUPANTS] of AGE_GROUPS: occupant_age_group; % the gender of the occupant.

% define variables that the solver can vary to find a solution
array[PATIENTS] of var bool: is_scheduled; % decision whether to treat a patient at all
array[PATIENTS] of var DAYS: patient_admission_day; % This is the surgery day and a patients stay ends after a given number of days
array[PATIENTS] of var ROOMS: patient_room_booking;
array[ROOMS, DAYS] of var AGE_GROUPS: age_of_youngest_patient;
array[ROOMS, DAYS] of var AGE_GROUPS: age_of_oldest_patient;

array[ROOMS, DAYS, PATIENTS] of var bool: is_patient_in_room;
constraint forall(r in ROOMS, d in DAYS, p in PATIENTS
    )(
        is_patient_in_room[r, d, p] == (is_scheduled[p]
        /\ patient_room_booking[p] == r
        /\ patient_admission_day[p] <= d
        /\ d <= patient_admission_day[p] + length_of_stay_patient[p] - 1)
    );

% H1: dont mix genders within a room
% if 2 different patients/occupants booked in the same room on the same day, that must mean they are the same gender
constraint forall(p1 in PATIENTS where is_scheduled[p1], p2 in PATIENTS where 
        p1 < p2 
        /\ is_scheduled[p2]
        /\ patient_room_booking[p1] == patient_room_booking[p2] 
        /\ patient_admission_day[p1] <= patient_admission_day[p2] + length_of_stay_patient[p2] - 1
        /\ patient_admission_day[p2] <= patient_admission_day[p1] + length_of_stay_patient[p1] - 1
    )(
        gender[p1] == gender[p2]
    );

constraint forall(p in PATIENTS where is_scheduled[p], o in OCCUPANTS where 
        patient_room_booking[p] == occupant_room_booking[o] 
        /\ patient_admission_day[p] <= occupant_length_of_stay[o] - 1
    )(
        gender[p] == occupant_gender[o]
    );

% H2: patients must be compatible with their room
% if a patient is booked, that means he must be compatible with the room he is booked in
constraint forall(p in PATIENTS where is_scheduled[p])(
        is_patient_compatible_with_room[p, patient_room_booking[p]]
    );


% H3: No overtime for surgeons
constraint forall(s in SURGEONS, d in DAYS)(
    sum(p in PATIENTS where is_scheduled[p] /\ surgeon_id[p] = s /\ patient_admission_day[p] = d)(
        surgery_duration[p]
    ) <= max_surgery_time[s, d]
);

% H5: All mandatory patients must be scheduled
constraint forall(p in PATIENTS where is_mandatory[p])(
        is_scheduled[p]
    );

% H6: surgery_day (admission) must be between release_day and due_day
constraint forall(p in PATIENTS where is_scheduled[p])(
        patient_admission_day[p] >= surgery_release_day[p]
    );
constraint forall(p in PATIENTS where is_mandatory[p])(
        patient_admission_day[p] <= surgery_due_day[p]
    );

% H7: room load can never be higher than the rooms capacity
array[ROOMS, DAYS] of int: room_capacity_per_day = array2d(ROOMS, DAYS, [room_capacity[r] - count([occupant_room_booking[o] == r /\ d <= occupant_length_of_stay[o] | o in OCCUPANTS]) | r in ROOMS, d in DAYS]);
constraint forall(r in ROOMS, d in DAYS)(
        count([is_patient_in_room[r, d, p] | p in PATIENTS]) <= room_capacity_per_day[r, d]
    );


% Objective
constraint forall(r in ROOMS, d in DAYS, p in PATIENTS where is_patient_in_room[r, d, p])(
    age_of_youngest_patient[r, d] <= age_group[p] /\ age_group[p] <= age_of_oldest_patient[r, d]
);
constraint forall(o in OCCUPANTS, r in ROOMS where r == occupant_room_booking[o], d in DAYS where d <= occupant_length_of_stay[o])(
    age_of_youngest_patient[r, d] <= occupant_age_group[o] /\ occupant_age_group[o] <= age_of_oldest_patient[r, d]
);
var int: total_unscheduled = count([not is_scheduled[p] | p in PATIENTS]);
var int: total_age_difference = sum(r in ROOMS, d in DAYS)(age_of_oldest_patient[r, d] - age_of_youngest_patient[r, d]);
var int: total_admission_delay = sum(p in PATIENTS where is_scheduled[p])(patient_admission_day[p] - surgery_release_day[p]);

solve minimize WEIGHT_UNSCHEDULED_OPTIONAL * total_unscheduled + WEIGHT_ROOM_MIXED_AGE * total_age_difference + WEIGHT_PATIENT_DELAY * total_admission_delay;
