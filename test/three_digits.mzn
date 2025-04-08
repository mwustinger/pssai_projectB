int: N; % number of allowed digits
set of int: allowed; % input digits

var allowed: a;
var allowed: b;
var allowed: c;

constraint
    a != b /\ b != c /\ a != c /\ a + b == c;

var int: total=a+b+c;

%solve satisfy; % for any solution
solve minimize total; % for minimizing the total

%output ["a=\(a), b=\(b), c=\(c)"];