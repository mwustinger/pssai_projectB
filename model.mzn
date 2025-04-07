% weights
int: WEIGHT_ROOM_MIXED_AGE;
int: WEIGHT_PATIENT_DELAY;
int: WEIGHT_UNSCHEDULED_OPTIONAL;

% define the sets of possible values of the constants
int: NUM_DAYS;
enum PATIENTS;
enum AGE_GROUPS;
enum GENDERS;
set of int: DAYS = 0..NUM_DAYS-1;
set of int: MINUTES = 0..1339;
enum SURGEONS;
enum ROOMS;

% define the arrays indexed by the sets
array[PATIENTS] of bool: is_mandatory; % a true/false array, where the ith value belongs to the ith patient
array[PATIENTS] of DAYS: surgery_release_day; % from this day surgery can be done
array[PATIENTS] of opt DAYS: surgery_due_day; % until this day surgery must be finished
array[PATIENTS] of SURGEONS: surgeon_id; % the surgeon who carries out the patient’s surgery
array[PATIENTS] of MINUTES: surgery_duration; %the expected duration of the patient’s surgery, which is assumed to always take place on the day of admission.
array[PATIENTS] of DAYS: length_of_stay; % duration of the hospitalization in days.
array[PATIENTS] of AGE_GROUPS: age_group; % the age group of the patient (e.g., infant, youth, adult, elder). The list of age groups is fully ordered.
array[PATIENTS] of GENDERS: gender; % the gender of the patient.
array[SURGEONS, DAYS] of MINUTES: max_surgery_time; % max duration a surgeon can work each day
array[ROOMS] of int: room_capacity; % max number of Patients allowed in here
array[PATIENTS, ROOMS] of bool: is_patient_compatible_with_room; % 2D matrix with patients as rows and rooms as cols

% define variables that the solver can vary to find a solution
array[PATIENTS] of var bool: is_scheduled; % decision whether to treat a patient at all
array[PATIENTS] of var DAYS: patient_admission_day; % This is the surgery day and a patients stay ends after a given number of days
array[PATIENTS] of var ROOMS: patient_room_booking; % only in combination with book_them_or_no is this even relevant

function array[PATIENTS] of bool: get_patients_of(ROOMS: r, DAYS: d) =
    let {
        array[PATIENTS] of bool: is_patient_in_room = 
            [is_scheduled[p] /\
                 patient_room_booking[p] = r /\
                 patient_admission_day[p] <= d /\
                 d <= patient_admission_day[p] + length_of_stay[p] | p in PATIENTS]
    } in is_patient_in_room;

/*
% H1: dont mix genders within a room
% if 2 different patients booked in the same room on the same day, that must mean they are the same gender
constraint forall(p1 in PATIENTS, p2 in PATIENTS where p1 < p2)(
        is_scheduled[p1] /\ is_scheduled[p2] /\
        patient_room_booking[p1] = patient_room_booking[p2] /\
        patient_admission_day[p1] <= patient_admission_day[p2] + length_of_stay[p2] -1 /\
        patient_admission_day[p2] <= patient_admission_day[p1] + length_of_stay[p1] -1
        -> gender[p1] = gender[p2]
    );

% H2: patients must be compatible with their room
% if a patient is booked, that means he must be compatible with the room he is booked in
constraint forall(p in PATIENTS)(
        is_scheduled[p] ->  is_patient_compatible_with_room[p, patient_room_booking[p]]
    );

% H3: No overtime for surgeons
function var int: get_surgeon_occupation_on_day(SURGEONS: s, DAYS: d) =
    sum([surgery_duration[p] * bool2int(
            is_scheduled[p] /\ surgeon_id[p] = s /\ patient_admission_day[p] = d
    ) | p in PATIENTS]);
constraint forall(s in SURGEONS. d in DAYS)(
        get_surgeon_occupation_on_day(s, d) <= max_surgery_time[s, d]
    );

% H5: All mandatory patients must be scheduled
constraint forall(p in PATIENTS)(is_mandatory[p] -> is_scheduled[p]);

% H6: surgery_day (admission) must be between has_release_day and has_due_day
% the booking days dont matter here, generally a patient can be booked the whole year, its just important that he gets
% surgery in the interval required
constraint forall(p in PATIENTS)(
        surgery_release_day[p] <= surgery_day[p]
    );
constraint forall(p in PATIENTS)(
        is_mandatory[p] -> surgery_day[p] <= surgery_due_day[p]
    );

% H7: room load can never be higher than the rooms capacity
constraint
    forall(r in ROOMS, d in DAYS)(
        sum(p in PATIENTS)(get_patients_of(r, d)[p]) <= room_capacity[r]
    );
    */


% Objective
function int: median(array[int] of int: data) =
    let {
        int: n = length(data),
        array[1..n] of int: sorted = sort(data)
    } in
    sorted[floor((n + 1) div 2)];

function var int: get_age_gap_on_day(ROOMS: r, DAYS: d) = 
    let {
        array[PATIENTS] of bool: is_patient_in_room = get_patients_of(r, d),
        int: med_ord = median([ord(AGE_GROUPS, age_group[p]) | p in PATIENTS where is_patient_in_room[p]])
    } in 
    sum([abs(ord(AGE_GROUPS, age_group[p])-med_ord) | p in PATIENTS where is_patient_in_room[p]]);

var int: num_mixed_age_groups = sum(r in ROOMS, d in DAYS)(get_age_gap_on_day(r, d));
var int: num_unscheduled = sum(p in PATIENTS)(1 - bool2int(is_scheduled[p]));

solve minimize WEIGHT_UNSCHEDULED_OPTIONAL * num_unscheduled + WEIGHT_ROOM_MIXED_AGE * num_mixed_age_groups;

/*
% define values that are a consequence of the decisions above
% in each room on each day, there can be a certain number of patients booked
% their count is the room load, which is constrained by the room's capacity
% define a function that computes the room load given a day
% do a list comprehension for all patients, get a boolean of whether they were booked and if theyre assigned to this
% room and their patient_booking_start and _end are before & after the current day
% then turn that boolean into an int, which you can then sum up to get the number of bookings in that room that day



% H5: mandatory patients must be scheduled!
% if a patient is mandatory, that means he is booked
constraint
    forall(p in PATIENTS)(
        is_mandatory[p] -> book_them_or_no[p]
    );



% surgeon can only do 1 surgery at a time
% i.e. any two patients can not have surgery with the same surgeon at the same time
% so if two patients are booked and their surgery intervals overlap, they cannot have the same surgeon
constraint
    forall(p1 in PATIENTS, p2 in PATIENTS where p1 < p2)(
        book_them_or_no[p1] /\ book_them_or_no[p2] /\  overlap(surgery_start[p1], surgery_end[p1], surgery_start[p2], surgery_end[p2])
        -> treating_surgeon[p1] != treating_surgeon[p2]
    );
*/