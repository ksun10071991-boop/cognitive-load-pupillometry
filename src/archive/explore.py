import pandas as pd

df = pd.read_parquet("colet.parquet")

# --- Question 1: who are the participants? ---
# .unique() lists each distinct value once; len() counts them
participants = df["Participant"].unique()
print("Number of participants:", len(participants))
print("Participant IDs:", sorted(participants))

# --- Question 2: how many tasks, and how many person-task blocks? ---
tasks = df["Task"].unique()
print("\nNumber of tasks:", len(tasks))
print("Task names:", sorted(tasks))

# a "block" = one unique (Participant, Task) pair = one labeled example
blocks = df.groupby(["Participant", "Task"]).size()
print("\nNumber of (participant, task) blocks:", len(blocks))
print("Rows per block — summary:")
print(blocks.describe())   # count, mean, min, max, etc.

# --- Question 3: what do the NASA scores look like? ---
# one score per block, so grab the first NASA value of each block
scores = df.groupby(["Participant", "Task"])["NASA_RTLX"].first()
print("\nNASA_RTLX score — summary:")
print(scores.describe())
print("\nAll the scores, sorted:")
print(sorted(scores.values))