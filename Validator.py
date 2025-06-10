# macOS Validation Cell
import subprocess
import os
from pathlib import Path
import json  # Für JSON-Validierung


def validate_all_solutions_macos(test_instance="./ihtc2024_test_dataset/test06.json", 
test_solution="./ihtc2024_test_dataset/test06_solution.json",
                                 verbose=False):
    pass


def validate_solution(instance_file, solution_file, verbose=True):
    if instance_file is None: 
        raise ValueError("No instance path supplied to Validator!")
    # Absoluter Pfad zum kompilierten Validator
    validator_path = os.path.abspath(os.path.join("validatorData", "IHTC_Validator"))

    # Überprüfe Testinstanz
    instance_path = os.path.abspath(instance_file)
    if not os.path.exists(instance_path):
        print(f"Error: Test instance not found at {instance_path}")
        return

    if verbose:
        print(f"Using test instance: {instance_path}")
        print("=" * 50)

        print(f"\nValidating {solution_file}:")
        print("-" * 30)
    solution_path = os.path.abspath(str(solution_file))

    # Überprüfe ob Lösungsdatei existiert
    if not os.path.exists(solution_path):
        print(f"Error: Solution file not found at {solution_path}")
        return

    # Überprüfe ob JSON valide ist
    try:
        with open(solution_path, 'r') as f:
            json.load(f)
        with open(instance_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in files: {e}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    try:
        cmd = [validator_path, instance_path, solution_path]
        if verbose:
            cmd.append("verbose")

        if verbose: print("Running command:", ' '.join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(validator_path)
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            if verbose:
                print("Errors:", result.stderr)
    except Exception as e:
        if verbose:
            print(f"Error validating {solution_file}: {e}")
        raise e
    if verbose:
        print("-" * 30)

# Ausführen der Validierung
#if __name__ == "__main__":
#    for i in range(1, 11):
#        index = f"{i:02d}"
#        validate_all_solutions_macos(
#            test_instance=f"./ihtc2024_test_dataset/test{index}.json",
#            test_solution=f"./ihtc2024_test_dataset/test{index}_solution_model3_simple.json"
#        )
#
#    # validate_all_solutions_macos(test_instance="./ihtc2024_test_dataset/test01.json",
#    #                                  test_solution="./ihtc2024_test_dataset/test01_solution_model3_simple.json")
