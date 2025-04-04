% define constants for array dimensions
int: NUM_PATIENTS; % n_p
int: NUM_AGE_GROUPS;
int: NUM_GENDERS;
int: NUM_DAYS;
int: NUM_SURGEONS;
int: NUM_ROOMS;

% define the sets of possible values of the constants
set of int: PATIENTS = 1..NUM_PATIENTS; % patients are numbered from 1 to n_p
set of int: AGE_GROUPS = 1..NUM_AGE_GROUPS;
set of int: GENDERS = 1..NUM_GENDERS;
set of int: DAYS = 1..NUM_DAYS;
set of int: HOURS = 1..24;
set of int: SURGEONS = 1..NUM_SURGEONS;
set of int: ROOMS = 1..NUM_ROOMS;

% define the arrays indexed by the sets
array[PATIENTS] of bool: is_mandatory; % a true/false array, where the ith value belongs to the ith patient
array[PATIENTS] of AGE_GROUPS: is_age_group;
array[PATIENTS] of GENDERS: has_gender;
array[PATIENTS] of DAYS: has_release_day; % from this day surgery can be done
array[PATIENTS] of DAYS: has_due_day; % until this day surgery must be finished
array[PATIENTS] of HOURS: surgery_duration;
array[SURGEONS] of HOURS: has_max_surgery_time; % TODO: this might be wrong, in Instance.py this is a list ???
array[ROOMS] of int: has_capacity; % max number of Patients allowed in here
array[PATIENTS, ROOMS] of bool: patient_compatible_with_room; % 2D matrix with patients as rows and rooms as cols

% define variables that the solver can vary to find a solution
array[PATIENTS] of var bool: book_them_or_no; % decision whether to treat a patient at all
array[PATIENTS] of var ROOMS: patient_room_booking; % only in combination with book_them_or_no is this even relevant
array[PATIENTS] of var DAY: patient_booking_start;
array[PATIENTS] of var DAY: patient_booking_end;
array[PATIENTS] of var DAYS: surgery_day;
array[PATIENTS] of var HOURS: surgery_start;
array[PATIENTS] of var HOURS: surgery_end;
array[PATIENTS] of var SURGEONS: treating_surgeon;
% the surgery day and time for the surgeons isn't a variable but rather a consequence of the variables defined here
% if a patient gets treated by a surgeon at a certain time, this surgeons timetable is thus defined

% define values that are a consequence of the decisions above
% in each room on each day, there can be a certain number of patients booked
% their count is the room load, which is constrained by the room's capacity
% define a function that computes the room load given a day
% do a list comprehension for all patients, get a boolean of whether they were booked and if theyre assigned to this
% room and their patient_booking_start and _end are before & after the current day
% then turn that boolean into an int, which you can then sum up to get the number of bookings in that room that day
function int: get_room_load_on_day(int: r, int: d) =
    sum([bool2int(
            book_them_or_no[p] /\ patient_room_booking[p] = r /\ patient_booking_start[p] <= d /\ d <= patient_booking_end[p]
    ) | p in PATIENTS])

% define a function for checking whether two ranges overlap
% you can use this to check whether any two bookings or surgeries overlap in time
function bool: overlap(int: start_a, int: end_a, int: start_b, int: end_b) = start_a < end_b /\ start_b < end_a;

% H1: dont mix genders within a room
% if 2 different patients booked in the same room on the same day, that must mean they are the same gender
constraint
    forall(p1 in PATIENTS, p2 in PATIENTS where p1 < p2)(
        book_them_or_no[p1] /\ book_them_or_no[p2] /\
        patient_room_booking[p1] = patient_room_booking[p2] /\
        overlap(patient_booking_start[p1], patient_booking_end[p1], patient_booking_start[p2], patient_booking_end[p2])
        -> has_gender[p1] = has_gender[p2]
    );

% H2: patients must be compatible with their room
% if a patient is booked, that means he must be compatible with the room he is booked in
constraint
    forall(p in PATIENTS)(
        book_them_or_no[p] ->  patient_compatible_with_room[p, patient_room_booking[p]]
    );

% H5: mandatory patients must be scheduled!
% if a patient is mandatory, that means he is booked
constraint
    forall(p in PATIENTS)(
        is_mandatory[p] -> book_them_or_no[p]
    );

% H7: room load can never be higher than the rooms capacity
constraint
    forall(r in ROOMS, d in DAYS)(
        get_room_load_on_day(r, d) <= has_capacity[r]
    );
