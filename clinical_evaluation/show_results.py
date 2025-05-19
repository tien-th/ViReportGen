import os
import json
from pathlib import Path # Using pathlib is often more convenient for paths

def calculate_totals_from_json_folder(folder_path_str: str) -> dict | None:
    """
    Calculates the sum of specific keys across multiple JSON files in a folder.

    Args:
        folder_path_str: The path (as a string) to the folder containing JSON files.

    Returns:
        A dictionary containing the aggregated totals for 'total_gt',
        'total_pred', and 'correct_predictions'.
        Returns None if the folder doesn't exist or isn't accessible.
    """
    folder_path = Path(folder_path_str)

    # Check if the provided path is a valid directory
    if not folder_path.is_dir():
        print(f"Error: Folder not found or is not a directory: {folder_path}")
        return None

    # Initialize counters
    total_gt_sum = 0
    total_pred_sum = 0
    correct_predictions_sum = 0
    files_processed = 0
    files_skipped = 0

    print(f"Scanning folder: {folder_path}")

    # Iterate through all files in the directory ending with .json
    for file_path in folder_path.glob('*.json'):
        print(f"--> Processing file: {file_path.name}")
        try:
            # Open and read the JSON file
            # Use encoding='utf-8' for better compatibility
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Safely get values using .get() with a default of 0
                # This prevents errors if a key is missing
                gt = data.get('total_gt', 0)
                pred = data.get('total_pred', 0)
                correct = data.get('correct_predictions', 0)

                # Optional: Add a check for numeric types if values might not be numbers
                if not all(isinstance(x, (int, float)) for x in [gt, pred, correct]):
                    print(f"    Warning: Non-numeric value found in {file_path.name}. Skipping this file.")
                    files_skipped += 1
                    continue # Skip to the next file

                # Add values to the running totals
                total_gt_sum += gt
                total_pred_sum += pred
                correct_predictions_sum += correct
                files_processed += 1

        except json.JSONDecodeError:
            print(f"    Error: Invalid JSON format in {file_path.name}. Skipping.")
            files_skipped += 1
        except IOError as e:
            print(f"    Error reading file {file_path.name}: {e}. Skipping.")
            files_skipped += 1
        except Exception as e: # Catch any other unexpected errors
            print(f"    An unexpected error occurred processing {file_path.name}: {e}. Skipping.")
            files_skipped += 1

    print(f"\nFinished scanning. Processed {files_processed} files, skipped {files_skipped} files.")

    # Return the results as a dictionary
    return {
        "total_gt": total_gt_sum,
        "total_pred": total_pred_sum,
        "correct_predictions": correct_predictions_sum
    }

# --- How to use the function ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your folder containing the JSON files
    json_folder = "/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/w_refactor/test"  # <--- CHANGE THIS PATH

    # Calculate the totals
    aggregated_totals = calculate_totals_from_json_folder(json_folder)

    # Print the results if calculation was successful
    if aggregated_totals is not None:
        print("\n--- Aggregated Totals ---")
        print(f"Total Ground Truth (total_gt): {aggregated_totals['total_gt']}")
        print(f"Total Predictions (total_pred): {aggregated_totals['total_pred']}")
        print(f"Total Correct Predictions (correct_predictions): {aggregated_totals['correct_predictions']}")

        # Optional: Calculate overall accuracy from totals
        total_pred = aggregated_totals['total_pred']
        if total_pred > 0:
            overall_accuracy = (aggregated_totals['correct_predictions'] / total_pred)
            print(f"\nOverall Accuracy (based on totals): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        elif aggregated_totals['correct_predictions'] > 0:
             print("\nOverall Accuracy: Cannot calculate (0 predictions but >0 correct? Check data).")
        else:
            print("\nOverall Accuracy: N/A (0 predictions)")
    else:
        print("Calculation could not be completed due to folder error.")