include "globals.mzn"; % required for alldifferent

array[1..8] of var 0..9: digits; % S,E,N,D,M,O,R,Y

constraint
    alldifferent(digits) /\
    digits[1] != 0 /\
    digits[5] != 0 /\
    1000 * digits[1] + 100 * digits[2] + 10 * digits[3] + digits[4] +
    1000 * digits[5] + 100 * digits[6] + 10 * digits[7] + digits[2] =
    10000 * digits[5] + 1000 * digits[6] + 100 * digits[3] + 10 * digits[2] + digits[8];

solve satisfy;
