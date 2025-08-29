import pyreadstat

# Load your SPSS files
exp1_df, _ = pyreadstat.read_sav("Masterfile Experiment 1.sav")
exp2_df, _ = pyreadstat.read_sav("Data_Experiment 2.sav")
neha_df, _ = pyreadstat.read_sav("Data_neha_revised_table.sav")

# Show first few rows to verify structure
print("Experiment 1:")
print(exp1_df.head())
print("\nExperiment 2:")
print(exp2_df.head())
print("\nNeha Table:")
print(neha_df.head())
