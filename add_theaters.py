import json
import os

folder_path = "./output/final_test"  # Replace with your folder path

for filename in os.listdir(folder_path):
    if filename.endswith(".json"): # and "test10" in filename:
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, "r") as file:
            data = json.load(file)

        for patient in data.get("patients", []):
            if patient.get("admission_day") != "none":
                patient["operating_theater"] = "t0"

        # Rename file if needed
        new_filename = filename.replace(".json_", "_") if ".json_" in filename else filename
        new_filepath = os.path.join(folder_path, new_filename)

        # Save updated data
        with open(new_filepath, "w") as file:
            json.dump(data, file, indent=2)

        # Remove old file if renamed
        if new_filepath != filepath:
            os.remove(filepath)

print("JSON files updated.")
