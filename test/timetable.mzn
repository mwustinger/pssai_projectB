int: NUM_CLASSES = 3; % there are 3 subjects
int: NUM_TIMES = 3; % there are 3 time slots
int: NUM_ROOMS = 2; % there are 2 rooms
int: NUM_TEACHERS = 2; % there are 2 teachers

% if NUM_CLASSES==3 then CLASSES = {1, 2, 3}
set of int: CLASSES = 1..NUM_CLASSES;
set of int: TIMES = 1..NUM_TIMES;
set of int: ROOMS = 1..NUM_ROOMS;
set of int: TEACHERS = 1..NUM_TEACHERS;

% teacher 1 teaches class 1 and 3, teacher 2 teaches class 2
array[CLASSES] of int: teacher_of_class = [1, 2, 1];

% Decision Variables
% create an array called time
% it is indexed by CLASSES ({1, 2, 3}) and each element in it is a decision variable
% that takes any value from the set TIMES
% time[1] is the time slot for class 1
% that means in time you assign a TIME to a CLASS
array[CLASSES] of var TIMES: time;
% and in room you assign a ROOM to a CLASS
array[CLASSES] of var ROOMS: room;

% no two clases in the same room at the same time
constraint
    forall(c1 in CLASSES, c2 in CLASSES where c1 < c2) (
        time[c1] = time[c2] /\ room[c1] = room[c2] -> false
    );

% a teacher cannot teach two classes at the same time
constraint
    forall(c1 in CLASSES, c2 in CLASSES where c1 < c2) (
        teacher_of_class[c1] = teacher_of_class[c2] /\ time[c1] = time[c2] -> false
    );

solve satisfy;

