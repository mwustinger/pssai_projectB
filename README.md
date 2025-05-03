# pssai_projectB
Download Minizinc
```
https://www.minizinc.org/

export PATH="$PATH:/home/martin/Downloads/squashfs-root/usr/bin"
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
# Create a single solution from a dataset instance
python .\Solver.py .\ihtc2024_test_dataset\test01.json  
python ./Solver.py ./ihtc2024_test_dataset/test01.json

# Validate a solution from a dataset instance and a solution
.\IHTP_Validator.exe .\ihtc2024_test_dataset\test01.json .\ihtc2024_test_dataset\test01_sol.json
./IHTP_Validator.exe ./ihtc2024_test_dataset/test01.json ./ihtc2024_test_dataset/test01_sol.json

# Visualize a saved solution for a dataset instance
python .\Solution.py .\ihtc2024_test_dataset\test01.json .\ihtc2024_test_solutions\sol_test01.json
python ./Solution.py ./ihtc2024_test_dataset/test01.json ./ihtc2024_test_solutions/sol_test01.json

```