# macOS Validation Cell
import subprocess
import os
from pathlib import Path
import json  # Für JSON-Validierung

def validate_all_solutions_macos(test_instance="ihtc2024_test_dataset/test06.json",
                                 test_solution="ihtc2024_test_dataset/test06_solution.json",
                                 verbose=False):
    # Absoluter Pfad zum kompilierten Validator
    validator_path = os.path.abspath(os.path.join("validatorData", "IHTC_Validator"))
    # results_dir = Path("data/results_old")
    solution_files = [test_solution]
    
    # Überprüfe Testinstanz
    instance_path = os.path.abspath(test_instance)
    if not os.path.exists(instance_path):
        print(f"Error: Test instance not found at {instance_path}")
        return
        
    print(f"Using test instance: {instance_path}")
    print(f"Found {len(solution_files)} solution files to validate\n")
    print("=" * 50)
    
    for solution_file in solution_files:
        print(f"\nValidating {solution_file}:")
        print("-" * 30)
        solution_path = os.path.abspath(str(solution_file))
        
        # Überprüfe ob Lösungsdatei existiert
        if not os.path.exists(solution_path):
            print(f"Error: Solution file not found at {solution_path}")
            continue
            
        # Überprüfe ob JSON valide ist
        try:
            with open(solution_path, 'r') as f:
                json.load(f)
            with open(instance_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in files: {e}")
            continue
        except Exception as e:
            print(f"Error reading files: {e}")
            continue
            
        try:
            cmd = [validator_path, instance_path, solution_path]
            if verbose:
                cmd.append("verbose")
            
            print("Running command:", ' '.join(cmd))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(validator_path)
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"Error validating {solution_file}: {e}")
        print("-" * 30)

# Ausführen der Validierung
if __name__ == "__main__":
    validate_all_solutions_macos()