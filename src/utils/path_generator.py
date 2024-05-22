import os

def generate_path(workspace_path):
    file_path = os.path.dirname(os.path.abspath(__file__))
    root_path = file_path.split("src")[0]
    return root_path + workspace_path

