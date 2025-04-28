# pssai_projectB
Download Minizinc
```
https://www.minizinc.org/
```

Create the venv
```
python -m venv venv
```
Start the venv
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Unrestricted
source venv/bin/activate
pip install pandas colour great_tables dataclasses-json minizinc

python .\Solution.py .\ihtc2024_test_dataset\test01.json .\ihtc2024_test_solutions\sol_test01.json
```

Run the script
```
python .\Solver.py .\ihtc2024_test_dataset\test01.json  
python ./Solver.py ./ihtc2024_test_dataset/test01.json

.\IHTP_Validator.exe .\ihtc2024_test_dataset\test01.json .\ihtc2024_test_dataset\test01_sol.json
python .\Solution.py .\ihtc2024_test_dataset\test01.json .\ihtc2024_test_solutions\sol_test01.json
```