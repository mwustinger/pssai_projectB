# pssai_projectB
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

```