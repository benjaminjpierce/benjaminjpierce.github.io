import plotly.graph_objects as go
import pandas as pd
import sankey as sk


# load data
data = sk.load_data_from_json("Artists.json")

# prepare dataframe (convert to decades)
data = sk.convert_decade_data(data)

# define common threshold
threshold = 75

# generate and show sankey diagrams for each combination
sk.generate_and_show_sankey(data, ['Nationality', 'decade'], {'decade': 0, 'Nationality': 'Nationality unknown'}, threshold, "Nationality vs. Decade")
sk.generate_and_show_sankey(data, ['Nationality', 'Gender'], {'Gender': 'Unknown', 'Nationality': 'Nationality unknown'}, threshold, "Nationality vs. Gender")
sk.generate_and_show_sankey(data, ['Gender', 'decade'], {'decade': 0, 'Gender': 'Unknown'}, threshold, "Gender vs. Decade")
sk.generate_and_show_sankey(data, ['Nationality', 'Gender', 'decade'], {'Nationality': 'Nationality unknown', 'Gender': 'Unknown', 'decade': 0}, threshold, "All Three Columns")



art = pd.DataFrame({'nationality':['A', 'A', 'B', 'C', 'D'],
                    'gender': ['M', 'M', 'M', 'F', 'M'],
                    'decade':['1930', '1950', '1940', '1930', '1940'],
                    'fourth test': ['1', '2', '2', '0', '3'],
                    'fifth': ['dingbat', 'doop', 'random', 'i did it', 'yup']})

print(art)
sk.generate_and_show_sankey(art, ['nationality', 'gender', 'decade', 'fourth test', 'fifth'])