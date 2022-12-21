from clearml import Dataset

def add_dataset(project, name, files_path):
	print('creating dataset in project {}, name {}, files_path {}'.format(project, name, files_path))
	ds = Dataset.create(dataset_project=project, dataset_name=name)
	ds.add_files(path=files_path)
	ds.upload()
	ds.finalize()
	return ds

def add_to_json(json_file, k, v):
	json_file[k] = v
	print('json_file {}'.format(json_file))
