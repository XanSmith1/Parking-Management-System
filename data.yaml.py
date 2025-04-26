
data_yaml_path = '/Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/DS.yaml'

data_yaml_content = f"""
train:   /Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/train
test: /Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/test
valid:  /Users/xandersmith/Desktop/UTL(ParkMngmnt)/full_DS/valid


nc: 2 
names: ['0:Vacant', '1:Occupied']  
"""


# Write the data.yaml file
with open(data_yaml_path, 'w') as file:
    file.write(data_yaml_content)

print(f"data.yaml file created at: {data_yaml_path}")
