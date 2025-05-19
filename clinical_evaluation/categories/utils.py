import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Set
from collections import defaultdict

path = "/home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed/pretrain_data/single_turn/align_test.json"

with open(path, "r") as f:
    data = json.load(f)

def convert_path(path):
    paths = path.split(".")[0].split("/")
    petct = paths[0]
    month = paths[1]
    day = paths[4].split("_")[1]
    patient = paths[4].split("_")[-1]

    full_path = f"{petct}_{month}_{day}_patient_{patient}_REPORT_patient_{patient}"
    return full_path

def lowercase_json_values(data):
  """
  Recursively traverses a Python object (list, dict, etc.) and converts
  all string values to lowercase.

  Args:
    data: The input Python object (often loaded from JSON,
          e.g., a list or dictionary).

  Returns:
    A new Python object with the same structure as the input, but with
    all string values lowercased. Non-string values are unchanged.
  """
  if isinstance(data, str):
    # Base case: If it's a string, lowercase it
    return data.lower()
  elif isinstance(data, dict):
    # If it's a dictionary, recursively process each value
    return {key: lowercase_json_values(value) for key, value in data.items()}
  elif isinstance(data, list):
    # If it's a list, recursively process each item
    return [lowercase_json_values(item) for item in data]
  else:
    # Otherwise (int, float, bool, None, etc.), return the value unchanged
    return data

def load_json(file_path: str) -> List[Dict[str, Any]]:
        """
        Load data từ file JSON và xử lý trường hợp dữ liệu là string
        
        Args:
            file_path (str): Đường dẫn tới file JSON
            
        Returns:
            List[Dict[str, Any]]: List các dictionary sau khi parse
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Nếu data là string (JSON string), parse nó
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # print(data)
                data = [data]  # Nếu không parse được, wrap nó trong list
                
        # Chuyển đổi thành list nếu cần
        if not isinstance(data, list):
            data = [data]
            
        return data


from pathlib import Path # More modern way to handle paths
def extract_unique_values_from_json_files(directory_path, target_key):
    """
    Loops through all JSON files in a directory, extracts values for a specific key
    from lists of dictionaries within those files, and returns a set of unique values.

    Args:
        directory_path (str or Path): The path to the directory containing JSON files.
        target_key (str): The dictionary key whose values need to be extracted.

    Returns:
        set: A set containing all unique values found for the target_key across
             all valid JSON files processed. Returns an empty set if no files
             are found or no values are extracted.
    """
    unique_values = set()
    dir_path = Path(directory_path) # Convert string path to Path object

    if not dir_path.is_dir():
        print(f"Error: Directory not found: {directory_path}")
        return unique_values # Return empty set

    # Use glob to find all files ending with .json in the directory
    for file_path in dir_path.glob('*.json'):
        try:
            # Open and load the JSON file, ensuring UTF-8 encoding for special chars

            data = load_json(file_path)

            # --- IMPORTANT: Adapt this part based on your JSON structure ---
            # Assuming the JSON file contains a LIST of dictionaries, like your example
            if isinstance(data, list):
                for item_dict in data:
                    # Check if the item is a dictionary and contains the target key
                    if isinstance(item_dict, dict) and target_key in item_dict:
                        value = item_dict[target_key].lower()
                        if value is not None: # Optional: Avoid adding None values
                            unique_values.add(value)
            # OPTIONAL: Handle case if JSON is a single dictionary directly
            # elif isinstance(data, dict) and target_key in data:
            #     value = data[target_key]
            #     if value is not None:
            #         unique_values.add(value)
            else:
                print(f"  Warning: Expected a list of dictionaries in {file_path.name}, but found {type(data)}. Skipping extraction for this file.")
            # --- End of structure-dependent part ---

        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from file: {file_path.name}. Skipping.")
        except FileNotFoundError:
             print(f"  Error: File not found during processing (shouldn't happen with glob unless race condition): {file_path.name}. Skipping.")
        except Exception as e:
            # Catch other potential errors (e.g., permissions)
            print(f"  Error processing file {file_path.name}: {e}. Skipping.")

    return unique_values


import re

def xu_ly_chuoi_vi_tri_nang_cao(chuoi_dau_vao, special_list):
  """
  Tách chuỗi mô tả vị trí dựa trên dấu phẩy và 'và', với điều kiện
  không tách 'và' nếu một trong các kết quả chỉ có một từ.

  Args:
    chuoi_dau_vao: Chuỗi đầu vào mô tả vị trí.

  Returns:
    Một list các chuỗi vị trí đã được tách và làm sạch theo quy tắc mới.
  """
  if not chuoi_dau_vao:
    return []
  if chuoi_dau_vao in special_list: 
      return [chuoi_dau_vao]
  # Bước 1: Tách chính bằng dấu phẩy và làm sạch
  if ';' in  chuoi_dau_vao: 
    comma_parts = re.split(r'\s*;\s*', chuoi_dau_vao.strip())
  else: 
    comma_parts = re.split(r'\s*,\s*', chuoi_dau_vao.strip())
  # Lọc bỏ các phần rỗng ban đầu
  cleaned_comma_parts = [part for part in comma_parts if part]

  final_result = []
  for segment in cleaned_comma_parts:
    # Bước 2 & 3 & 4: Xử lý tách phụ bằng " và " với điều kiện
    if " và " in segment and segment not in special_list:
      # Tách thử bằng " và "
      potential_sub_parts = [p.strip() for p in segment.split(" và ") if p.strip()]

      # Kiểm tra điều kiện "một từ"
      should_keep_original = False
      if len(potential_sub_parts) > 1: # Chỉ kiểm tra nếu thực sự có thể tách
        for sub_part in potential_sub_parts:
          # Đếm số từ trong phần con (tách bằng khoảng trắng)
          word_count = len(sub_part.split())
          if word_count <= 1:
            should_keep_original = True # Tìm thấy phần con chỉ có 1 từ -> không tách

      if should_keep_original:
        # Giữ nguyên segment gốc (không tách bằng " và ")
        final_result.append(segment)
      elif len(potential_sub_parts) > 1 :
         # Tách hợp lệ (tất cả các phần con > 1 từ)
         final_result.extend(potential_sub_parts)
      else:
          # Trường hợp chỉ có 1 phần tử sau khi split " và " hoặc không split được
          final_result.append(segment)

    else:
      # Không chứa " và ", thêm trực tiếp segment
      final_result.append(segment)

  # Bước 5: Trả về kết quả cuối cùng (đã lọc rỗng lần nữa cho chắc)
  return [part for part in final_result if part]

from typing import Set, Tuple

def remove_and_report_single_word_strings(input_set: Set[str]) -> Set[str]:
    """
    Loại bỏ các chuỗi chỉ có một từ khỏi một tập hợp (set) và in ra các chuỗi đã bị loại bỏ.

    Hàm này duyệt qua tập hợp đầu vào, xác định các chuỗi chỉ chứa đúng một từ
    (sau khi loại bỏ khoảng trắng thừa ở đầu/cuối và tách bằng khoảng trắng).
    Các chuỗi một từ này sẽ được thu thập và in ra, đồng thời bị loại bỏ
    khỏi tập hợp kết quả được trả về.

    Args:
        input_set: Một tập hợp (set) các chuỗi đầu vào.

    Returns:
        Một tập hợp (set) mới chỉ chứa các chuỗi từ tập hợp đầu vào
        mà KHÔNG PHẢI là chuỗi một từ.
    """
    if not isinstance(input_set, set):
        print("Warning: Input is not a set. Attempting to process.")
        try:
            input_set = set(input_set)
        except TypeError:
            raise TypeError("Input must be convertible to a set of strings.")

    strings_to_remove = set()
    strings_to_keep = set()

    for item in input_set:
        # Đảm bảo xử lý string, bỏ qua các kiểu khác nếu có
        if isinstance(item, str):
            # item.split() tách chuỗi thành list các từ dựa trên khoảng trắng.
            # Nó tự động xử lý khoảng trắng thừa và trả về list rỗng cho chuỗi rỗng hoặc chỉ có khoảng trắng.
            words = item.split()
            if len(words) == 1:
                strings_to_remove.add(item) # Thêm vào danh sách sẽ bị loại bỏ
            else:
                # Giữ lại các chuỗi có 0 từ (chuỗi rỗng/khoảng trắng) hoặc nhiều hơn 1 từ
                strings_to_keep.add(item)
        else:
            # Nếu không phải string, giữ lại nó (vì nó không phải là "chuỗi một từ")
            strings_to_keep.add(item)

    # ----- In ra các chuỗi đã bị loại bỏ -----
    if strings_to_remove:
        print("--- Removed single-word strings ---")
        # Sắp xếp để in ra có thứ tự (tùy chọn, vì set vốn không có thứ tự)
        for removed_str in sorted(list(strings_to_remove)):
            print(f"- '{removed_str}'")
        print("---------------------------------")
    else:
        print("--- No single-word strings found to remove ---")

    return strings_to_keep

from typing import Set, Any

def write_set_to_file(data_set: Set[Any], filename: str) -> bool:
    """
    Writes all elements of a set to a text file, each on a new line.

    The elements are converted to strings before writing.
    For consistent output, the elements are sorted alphabetically after
    being converted to strings.

    Args:
        data_set: The set containing the data to write. Elements can be of
                  any type that can be converted to a string.
        filename: The name (including path if necessary) of the file to write to.
                  The file will be created if it doesn't exist, or overwritten
                  if it does exist.

    Returns:
        True if writing was successful, False otherwise.
    """
    # --- Input Validation ---
    if not isinstance(data_set, set):
        print(f"Error: Input 'data_set' is not a set.")
        return False
    if not isinstance(filename, str) or not filename.strip():
        # Check if filename is a non-empty string
        print("Error: Invalid or empty filename provided.")
        return False

    try:
        # --- Prepare Data for Writing ---
        # 1. Convert all set elements to strings to ensure writability
        #    and enable consistent sorting even with mixed types (like numbers).
        # 2. Sort the list of strings alphabetically.
        # 3. Create a list where each string ends with a newline character ('\n').
        lines_to_write = [str(item) + '\n' for item in sorted([str(x) for x in data_set])]

        # --- Write to File ---
        # Use 'with open' ensures the file is properly closed afterwards,
        # even if errors occur.
        # Mode 'w' means write (creates file or truncates existing file).
        # 'encoding='utf-8'' is crucial for handling various characters reliably.
        with open(filename, 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines_to_write) # Efficiently writes a list of strings

        print(f"Successfully wrote {len(data_set)} items to '{filename}'")
        return True

    except IOError as e:
        # Handle potential errors during file operations (e.g., permission denied)
        print(f"Error writing to file '{filename}': {e}")
        return False
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        return False


import os
from typing import Set

def read_set_from_file(filename: str) -> Set[str]:
    """
    Reads lines from a text file and returns them as a set of strings.

    Each line in the file is treated as a potential element for the set.
    Leading/trailing whitespace (including newline characters) is removed
    from each line before adding it to the set. Empty lines after stripping
    will result in an empty string element "" in the set if present.
    Duplicate lines in the file will result in only one entry in the returned set.

    Args:
        filename: The name (including path if necessary) of the file to read from.

    Returns:
        A set containing the unique, stripped lines from the file as strings.
        Returns an empty set if the file doesn't exist, cannot be read,
        or an error occurs.
    """
    # --- Input Validation ---
    if not isinstance(filename, str) or not filename.strip():
        print("Error: Invalid or empty filename provided.")
        return set() # Return empty set for invalid filename

    # --- Check if file exists before trying to open ---
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        return set() # Return empty set if file doesn't exist

    # --- Read from File ---
    read_elements = set()
    try:
        # Use 'with open' for automatic file closing
        # Mode 'r' for reading
        # encoding='utf-8' is important for broad character support
        with open(filename, 'r', encoding='utf-8') as f_in:
            # Use a set comprehension for concise reading and stripping
            # line.strip() removes leading/trailing whitespace and newlines
            read_elements = {line.strip() for line in f_in}

        print(f"Successfully read {len(read_elements)} unique elements from '{filename}'")
        return read_elements

    except IOError as e:
        # Handles errors like permission denied
        print(f"Error reading file '{filename}': {e}")
        return set() # Return empty set on IO error
    except UnicodeDecodeError as e:
        # Handle cases where the file encoding is not UTF-8
        print(f"Error decoding file '{filename}' (likely not UTF-8): {e}")
        print("Tip: Ensure the file is saved with UTF-8 encoding.")
        return set() # Return empty set on decoding error
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return set() # Return empty set on other errors


def remove_x_at_start(text, x):
    # Loại bỏ chuỗi x nếu nó xuất hiện ở đầu chuỗi text
    text = re.sub(r'^\s*' + re.escape(x) + r'\s+', '', text)
    return text