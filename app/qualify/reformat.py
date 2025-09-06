import json

CACHE_FILE = "app/qualify/cache.json"


if __name__ == "__main__":
    with open(CACHE_FILE, "r") as file:
        data = json.load(file)

    if all(x not in data for x in ('good', 'bad')):
        exit(0)

    new_data = {}
    for project_path, module_name in data['good']:
        project_name = project_path.split('/')[-1]
        if project_name in new_data:
            new_data[project_name][module_name] = 1
        else:
            new_data[project_name] = {
                '__ignored__': [],
                module_name: 1
            }

    for project_path, module_name in data['bad']:
        project_name = project_path.split('/')[-1]
        if project_name in new_data:
            new_data[project_name][module_name] = 0
        else:
            new_data[project_name] = {
                '__ignored__': [],
                module_name: 0
            }

    with open(CACHE_FILE, "w") as file:
        json.dump(new_data, file, indent=4)
