import subprocess
import os
import csv
import re

instance_dir = "./ihtc2024_test_dataset/"
solution_dir = "./output/final_test/"
validator_exe = "./IHTC_Validator.exe"

output_csv = "validation_results.csv"

# Headers to extract from the validator output
violation_keys = [
    "RoomGenderMix",
    "PatientRoomCompatibility",
    "SurgeonOvertime",
    "OperatingTheaterOvertime",
    "MandatoryUnscheduledPatients",
    "AdmissionDay",
    "RoomCapacity",
    "NursePresence",
    "UncoveredRoom",
    "Total violations"
]

cost_keys = [
    "RoomAgeMix",
    "RoomSkillLevel",
    "ContinuityOfCare",
    "ExcessiveNurseWorkload",
    "OpenOperatingTheater",
    "SurgeonTransfer",
    "PatientDelay",
    "ElectiveUnscheduledPatients",
    "Total cost"
]

all_keys = ["instance"] + violation_keys + cost_keys

# Prepare CSV file
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_keys)
    writer.writeheader()

    # Loop over each solution file
    for fname in os.listdir(solution_dir):
        if not fname.endswith(".json"):
            continue

        solution_path = os.path.join(solution_dir, fname)

        # Extract instance file name (e.g. test01.json)
        match = re.match(r"(test\d+)_solution", fname)
        if not match:
            continue
        instance_name = match.group(1) + ".json"
        instance_path = os.path.join(instance_dir, instance_name)

        # Run validator
        result = subprocess.run(
            ["wine", validator_exe, instance_path, solution_path],
            capture_output=True,
            text=True
        )

        output = result.stdout
        row = {"instance": fname}

        # Extract values using regex
        for key in violation_keys + cost_keys:
            pattern = rf"{re.escape(key)}[.\s]*([0-9]+)"
            match = re.search(pattern, output)
            row[key] = int(match.group(1)) if match else None

        writer.writerow(row)

print(f"✅ Results saved to {output_csv}")
