import os
import json

from datetime import datetime
import argparse
import pickle

from clearml import Task
from clearml import Model
from clearml import OutputModel

from utils import add_dataset
from utils import add_to_json


def import2standalone(args):
    '''
    Import all experiment tasks & these tasks will have its newly generated id by standalone ClearML.
    Import models and datasets generated and used by the experiment tasks respectively.
    '''
    import_info = {}
    add_to_json(import_info, 'importer', '')
    add_to_json(import_info, 'imported_on', str(datetime.now()))

    # get deploy info
    deploy_info = get_deploy_info(args.deploy_info)

    # get tasks to import
    tasks_to_import = deploy_info['tasks']
    datasets_details = deploy_info['datasets']

    # import task, model(s) and dataset(s)
    for t in tasks_to_import:
        print('t {}'.format(t))
        t_id = t['id']
        t_models = t['models']
        t_datasets = t['datasets']

        # import task
        imported_task_id = import_experiment_task(
            t_id,
            args.tasks_folder,
            import_info)
        print('imported_task_id {}'.format(imported_task_id))

        # import model(s) to task
        uploaded_url = import_models_for_task(
            imported_task_id,
            t_models,
            args.models_folder,
            args.output_uri,
            import_info)
        print('model uploaded to url {}'.format(uploaded_url))

        # import dataset(s) and link to task
        import_datasets_for_task(
            imported_task_id,
            t_datasets,
            datasets_details,
            args.datasets_folder,
            import_info)

    # write import info to file
    with open('import.json', 'w') as f:
        f.write(json.dumps(import_info))

    print('Importing completed on {} ......'.format(datetime.now()))

def import_experiment_task(task_id, tasks_folder, import_info):
    '''
    Load task object.
    Import task object to ClearML.
    Write old_id - new_id to import.json.
    '''
    # read from file
    t_path = os.path.join(tasks_folder, task_id)
    f = open(t_path, "rb")
    t_obj = pickle.load(f)

    # import to standalone ClearML
    new_t_id = Task.import_task(t_obj).id

    # add to json
    values = { new_t_id:
        {
            'development_task_id': task_id,
            'standalone_task_id': new_t_id
        }
    }
    add_to_json(import_info, 'imported_task', values)

    return new_t_id

def import_models_for_task(
    imported_task_id,
    t_models,
    models_folder,
    output_uri,
    import_info):
    '''
    Import model object to standalone experiment task.
    Write old_id - new_id to import.json.
    '''
    print('t_models {}'.format(t_models))
    for m in t_models:
        m_name = m['name']
        m_id = m['model']

        # load model and attach to task
        t = Task.get_task(imported_task_id)
        output = OutputModel(t)
        uploaded_url = output.update_weights(
            weights_filename=os.path.join(models_folder, m_id + '.pt'),
            target_filename=m_name,
            upload_uri=output_uri,
            auto_delete_file=False)

        # get uploaded model id
        new_model_id = output.id
        print('uploaded model id {}'.format(new_model_id))

        # add to json
        values = { new_model_id:
            {
                'development_model_id': m_id,
                'standalone_model_id': new_model_id
            }
        }
        add_to_json(import_info, 'imported_model', values)
    return uploaded_url

def import_datasets_for_task(
    imported_task_id,
    t_datasets,
    datasets_details,
    datasets_folder,
    import_info):
    '''
    Import model object to standalone experiment task.
    Write old_id - new_id to import.json.
    '''
    print('importing datasets {} for task {}'.format(t_datasets, imported_task_id))
    new_ds_ids = []
    for ds_id in t_datasets:
        # get dataset project and name used in dev env
        proj = datasets_details[ds_id]['project']
        name = datasets_details[ds_id]['name']

        # create dataset in standalone
        # TODO: check for existing dataset to avoid overwritten,
        # TODO: raise for deconflicting if there is before continue
        files_path = os.path.join(datasets_folder, ds_id)
        new_ds = add_dataset(proj, name, files_path)
        new_ds_id = new_ds.id
        new_ds_ids.append(new_ds_id)
        print('ds {} added'.format(new_ds_id))

    # replace dataset id in imported task
    t = Task.get_task(imported_task_id)
    t_config = t.export_task()
    t_config['runtime']['datasets'] = new_ds_ids
    t.update_task(t_config)
    # print('dataset id {}'.format(t_config['runtime']['datasets']))

    # TODO: need to update orig_datasets.Dataset_mnist/1.0.0?

    # add to json
    values = { new_ds_id:
        {
            'development_dataset_id': ds_id,
            'standalone_dataset_id': new_ds_id
        }
    }
    add_to_json(import_info, 'imported_dataset', values)
    return ds_id

def get_args():
    parser = argparse.ArgumentParser(description = 'importing')
    parser.add_argument('--output-uri', default='http://192.168.50.1:8081',
        help='')
    parser.add_argument('--deploy-info', default='deploy.json',
        help='This is the json file that contains tasks, models and datasets to be imported to the standalone ClearML.')
    parser.add_argument('--tasks-folder', default='tasks',
        help='This is where the tasks are.')
    parser.add_argument('--models-folder', default='models',
        help='This is where the models are.')
    parser.add_argument('--datasets-folder', default='datasets',
        help='This is where the datasets are.')
    return parser.parse_args()

def get_deploy_info(deploy_info):
    f = open(deploy_info)
    return json.load(f)

# def main(args):
#   import_info = {}
#   add_to_json(import_info, 'importer', '')
#   add_to_json(import_info, 'imported_on', str(datetime.now()))

#   # get deploy info
#   deploy_info = get_deploy_info(args.deploy_info)

#   # get tasks to import
#   tasks_to_import = deploy_info['tasks']
#   datasets_details = deploy_info['datasets']
#   import2standalone(
#       tasks_to_import, 
#       datasets_details,
#       args.tasks_folder, 
#       args.models_folder, 
#       args.datasets_folder, 
#       args.output_uri, 
#       import_info)

#   # write import info to file
#   with open('import.json', 'w') as f:
#       f.write(json.dumps(import_info))


if __name__ == '__main__':
    # log this deployment too
    Task.init(project_name='Deploying2Standalone', task_name=str(datetime.now().date()), output_uri=True)
    args = get_args()
    import2standalone(args)
