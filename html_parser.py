import csv

# Load the page number and link CSV
page_number_and_link = {}
with open("page_numbers_and_links.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        page_number_and_link[row['Page Number']] = f"https://childrens-oncology-handbook.netlify.app/{row['Link']}"

# Load the handbook CSV to append links
handbook_data = []
with open("English_COG_Family_Handbook.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        link = page_number_and_link.get(row['Labeled Page Number'], "")
        row['Page Link'] = link
        handbook_data.append(row)

# Define the output file path
output_file_path = "Updated_English_COG_Family_Handbook.csv"

# Write the updated handbook data with page links to a new CSV
with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = reader.fieldnames + ['Page Link']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in handbook_data:
        writer.writerow(row)


