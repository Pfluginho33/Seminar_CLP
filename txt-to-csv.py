import csv

def convert_txt_to_csv(txt_file_path, csv_file_path):
    with open(txt_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in txt_file:
            row = line.strip().split()
            writer.writerow(row)

# Basispfad für die Dateien
base_path = '/Users/marcelpflugfelder/Documents/02_Studium/Master/Semester 4/07_Seminar/files/pmed'

# Schleife, die von 6 bis 40 läuft
for i in range(1, 41):
    txt_file_path = f"{base_path}{i}.txt"
    csv_file_path = f"{base_path}{i}.csv"
    convert_txt_to_csv(txt_file_path, csv_file_path)
