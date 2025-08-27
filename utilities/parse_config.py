import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current file's directory
project_dir = os.path.dirname(current_dir)  # Get the absolute path of the project's directory

file_name = "parameters.json"
config_file_path = os.path.join(project_dir, "utilities", file_name)
with open(config_file_path, 'r') as f:
    parameters = json.load(f)
    parameters['project_dir'] = project_dir
    parameters['data_dir'] = os.path.join(project_dir, parameters['data_dir'])
    path=os.path.join(parameters['data_dir'], 'raw', 'data')
    parameters['subject_names_train'] = [folder for folder in os.listdir(path)
                                   if os.path.isdir(os.path.join(path, folder))]


