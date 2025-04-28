import os
import pandas as pd

# Directories
input_dir = 'metrics'  # Directory where metrics are saved
output_file = 'top_5_ClusteringCoefficient.csv'

# Initialize results dictionary
results = {f'sub-{i:02d}': {"task-rest_run-1": None, "task-rest_run-2": None, "sleep": []} for i in range(1, 34)}

# Process each subject's folder
for subj in range(1, 34):
    subj_id = f'sub-{subj:02d}'
    subj_dir = os.path.join(input_dir, subj_id)

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

            # Sort by weighted degree centrality for positive graph
            top_5_nodes = df.sort_values("ClusteringCoefficient", ascending=False).head(5)["Node"].tolist()

            if state == "sleep":
                # Collect sleep top nodes for later combination
                results[subj_id][state].append(top_5_nodes)
            else:
                # Store top 5 nodes for task-rest states
                results[subj_id][state] = top_5_nodes

# Combine sleep state top nodes and prepare results table
rows = []
for subj_id, states in results.items():
    row = {"Subject": subj_id}

    for state in ["task-rest_run-1", "task-rest_run-2", "sleep"]:
        if state == "sleep":
            # Combine all sleep nodes
            all_sleep_nodes = [node for sleep_nodes in states[state] for node in sleep_nodes]
            if all_sleep_nodes:
                top_5_sleep = sorted(set(all_sleep_nodes), key=all_sleep_nodes.count, reverse=True)[:5]
                row[state] = ",".join(map(str, top_5_sleep))
            else:
                row[state] = "nan"
        else:
            # Format top 5 nodes for task-rest states
            if states[state]:
                row[state] = "/".join(map(str, states[state]))
            else:
                row[state] = "nan"

    rows.append(row)

# Save results to CSV
top_5_df = pd.DataFrame(rows)
top_5_df.to_csv(output_file, index=False)

print(f"Top 5 weighted degree centrality nodes saved to {output_file}")
