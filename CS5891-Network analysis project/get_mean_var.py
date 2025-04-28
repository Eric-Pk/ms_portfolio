import os
import pandas as pd
import numpy as np

input_dir = 'metrics'  # Adjust to match your metrics output directory
output_file = 'clustering_coefficient_summary.csv'

# Initialize results dictionary
results = {f'sub-{i:02d}': {"task-rest_run-1": None, "task-rest_run-2": None, "sleep": []} for i in range(1, 34)}

# Process each subject's folder
for subj in range(1, 34):
    subj_id = f'sub-{subj:02d}'
    in_dir = os.path.join(input_dir, subj_id)
    if not os.path.exists(subj_dir):
        continue

    for filename in os.listdir(subj_dir):
        if filename.endswith("_node_metrics.csv"):
            file_path = os.path.join(subj_dir, filename)

            # Identify the state from the filename
            if "task-rest_run-1" in filename:
                state = "task-rest_run-1"
            elif "task-rest_run-2" in filename:
                state = "task-rest_run-2"
            elif "sleep" in filename:
                state = "sleep"
            else:
                continue  # Skip unknown states

            # Load the CSV file
            df = pd.read_csv(file_path)

            # Calculate mean and variance of degree centrality for positive graph
            mean_degree = df['ClusteringCoefficient'].mean()
            var_degree = df['ClusteringCoefficient'].var()

            if state == "sleep":
                # Append sleep values for averaging later
                results[subj_id][state].append((mean_degree, var_degree))
            else:
                # Store single value for task-rest_run-1 and task-rest_run-2
                results[subj_id][state] = (mean_degree, var_degree)

# Prepare the results table
rows = []
for subj_id, states in results.items():
    row = {"Subject": subj_id}

    for state in ["task-rest_run-1", "task-rest_run-2", "sleep"]:
        if state == "sleep":
            # Average multiple sleep values if they exist
            sleep_values = states["sleep"]
            if sleep_values:
                mean_values = np.mean([v[0] for v in sleep_values])
                var_values = np.mean([v[1] for v in sleep_values])
                row[state] = f"{mean_values:.4f}({var_values:.4f})"
            else:
                row[state] = "nan(nan)"
        else:
            # Use single value for task-rest_run-1 and task-rest_run-2
            value = states[state]
            if value:
                row[state] = f"{value[0]:.4f}({value[1]:.4f})"
            else:
                row[state] = "nan(nan)"

    rows.append(row)

# Convert to DataFrame and save
summary_df = pd.DataFrame(rows)
summary_df.to_csv(output_file, index=False)

print(f"Summary table saved to {output_file}")
