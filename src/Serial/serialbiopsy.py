import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Count number of biopsies per patient
biopsy_counts = df.groupby("patient_id").size()

# Identify patients with serial biopsies (>1 biopsy)
serial_patients = biopsy_counts[biopsy_counts > 1]

# Number of patients with serial biopsies
n_serial_patients = serial_patients.shape[0]

# Number of biopsy episodes from serial patients
n_serial_biopsies = df[df["patient_id"].isin(serial_patients.index)].shape[0]

print("Number of patients with serial biopsies:", n_serial_patients)
print("Number of biopsy episodes from serial patients:", n_serial_biopsies)