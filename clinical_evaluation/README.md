### Extract structured lesion information from the LLM output

Use extract_value_gpt.py to extract structured lesion information from the LLM output by this command:
```bash
python extract_value_gpt.py --input_file <path_to_llm_output_file.json> --output_folder <path_to_output_folder>
```

### Evaluate the predictions

use evaluate.py to evaluate the predictions by this command:

# fields: type of lesion, position of lesion, FDG of lesion
Revise evaluate.py to change the criteria that you want to evaluate
```python
fields = [
            'hinh_dang', # type of lesion
            'vi_tri', # position of lesion
            #'xam_lan', # invasion of lesion
            'tang_chuyen_hoa' # FDG of lesion
            ] 
```


```bash
python evaluate.py --input_folder <path_to_llm_output_folder> --output_folder <path_to_output_folder> 
```

For example: 
```bash
python evaluate.py --input_folder ./example_data4evaluate --output_folder ./results/example_data4evaluate/type+position
```

- And you will get the details of each case in the output folder
- Number of lesion, number of total predictions, and number of correct predictions by each criteria will be displayed in the terminal



