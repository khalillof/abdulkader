# Optimized Scripts Documentation

## Fixes Applied:
1. **2_Knowledge_Graph_Construction_Optimized.py**  
   - Fixed invalid `%run` command by replacing with `exec(open("1_Kaggle_Data_Processing.py").read())`.

2. **3_GNN_Model_Architecture_Optimized.py**  
   - Fixed invalid `%run` command using `exec(open("1_Kaggle_Data_Processing.py").read())`.

3. **4_Training_and_Evaluation_Optimized.py**  
   - No changes made. Ensure PyTorch is installed properly before running.

4. **5_Results_Analysis_Optimized.py**  
   - No changes made. Fix memory issues by ensuring sufficient system resources.

5. **6_Knowledge_Graph_Construction_from_MIMIC_Optimized.py**  
   - No changes made. Ensure the correct NCCL library is installed for PyTorch.

6. **7_Experimental_Configuration_Optimized.py**  
   - Fixed unterminated triple-quoted string.

## How to Run the Scripts:

1. Ensure Python 3 is installed:  
   ```sh
   python3 --version
   ```

2. Install necessary dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
   *(You may need to manually install PyTorch and NCCL if required.)*

3. Run each script individually:  
   ```sh
   python3 script_name.py
   ```
   Example:  
   ```sh
   python3 1_Kaggle_Data_Processing_Optimized.py
   ```

4. For debugging, use logging:  
   ```sh
   tail -f log_file.log
   ```

Ensure all data files are in the correct directory before running the scripts.
