import os
import json

from datetime import datetime
import argparse
import pickle
import requests

from clearml import Task
from clearml import Model
from clearml import Dataset

from utils import add_to_json


def export_from_development(args):
    '''
    This method will export the experiment from the dev env.
    '''
    deploy_info = {}

    # write exporter info
    add_to_json(deploy_info, 'developer', '')
    add_to_json(deploy_info, 'exported_on', str(datetime.now()))

    # start exporting
    deployed_models = export_models(args.serving_service_id, deploy_info)
    experiment_task_ids = export_experiment_tasks(deployed_models, deploy_info)
    dataset_name, dest = export_datasets(experiment_task_ids, deploy_info)
    # export_basedockers(deploy_info)
    # export_preprocess(deploy_info)

    # write deploy info to file
    with open('deploy.json', 'w') as f:
        f.write(json.dumps(deploy_info))

def get_model_info(endpoints):
    models = []
    for endpoint, ep_info in endpoints.items():
        # print('ep_info {}'.format(ep_info))
        model_id = ep_info['model_id']
        model_url = Model(model_id).url
        models.append((model_id, model_url))
    # print('models {}'.format(models))
    return models

def export_models(serving_svs_id, deploy_info):
    '''
    Retrieve endpoint(s) from Serving Service. 
    Obtain model id and url for each endpoints.
    Download the model from ClearML Server and save to local drive. 
    '''
    add_to_json(deploy_info, 'serving_id', serving_svs_id)
    serving_svs = Task.get_task(serving_svs_id)

    endpoints = serving_svs.get_configuration_object_as_dict('endpoints')
    # write_to_manifest('endpoints', endpoints.items()[0])

    # save model to local drive
    models = get_model_info(endpoints)
    if not os.path.exists('models'): os.mkdir('models')
    for model_id, model_url in models:
        add_to_json(deploy_info, 'model', { 'id': model_id, 'url': model_url } )
        r = requests.get(model_url)
        model_path = os.path.join('models', model_id + '.pt')
        with open(model_path, "wb") as f:
            f.write(r.content)

    return models

def export_experiment_tasks(models, deploy_info):
    '''
    Get experiment task that generated the models.
    Output the experiment task(s) as pickle object(s).  
    '''
    experiment_task_ids = []
    experiment_tasks = []
    for model_id, _ in models:
        # get experiment task id
        experiment_task_id = Model(model_id).task
        experiment_task = Task.get_task(experiment_task_id).export_task()
        # print('experiment_task {}'.format(experiment_task))
        # print('model name {}'.format(experiment_task['models']['output']))
        task_models = experiment_task['models']['output']
        experiment_task_ids.append(experiment_task_id)

        # save task object as pickle
        if not os.path.exists('tasks'): os.mkdir('tasks')
        task_folder = os.path.join('tasks', experiment_task_id)
        with open(task_folder, 'wb') as f:
            pickle.dump(experiment_task, f)

        # get datasets info
        dataset_ids = experiment_task['runtime']['datasets']

        # deploy info
        experiment_tasks.append(
            { 'id': experiment_task_id,
              'models': task_models,
              'datasets': dataset_ids
            })

    # write to json
    add_to_json(deploy_info, 'tasks', experiment_tasks)

    return experiment_task_ids

def export_datasets(experiment_task_ids, deploy_info):
    '''
    Retrieve datasets used in experiment tasks.
    Download the datasets with its url.
    '''
    for task_id in experiment_task_ids:
        # get all the datasets used in this experiment task
        experiment_task = Task.get_task(task_id).export_task()
        dataset_ids = experiment_task['runtime']['datasets']
        print('dataset_ids {} used in task {}'.format(dataset_ids, task_id))

        # save datasets to local drive
        for dataset_id in dataset_ids:
            # get dataset task, proj, name
            dataset_task = Task.get_task(dataset_id)
            dataset_name = dataset_task.export_task()['name']
            dataset_proj = dataset_task.get_project_name().split('/')[0].strip()
            print('dataset proj {}, name {}, id {}'.format(dataset_proj, dataset_name, dataset_id))
            # print(dataset_task.export_task())

            # download dataset to destinated folder
            ds = Dataset.get(dataset_id=dataset_id)
            if not ds.is_final: ds.finalize()  # only finalize dataset can be copied
            dest = os.path.join(
                os.getcwd(),
                'datasets',
                dataset_id)
            ds.get_mutable_local_copy(target_folder=dest)

            # save dataset task as pickle
            if not os.path.exists('datasets_1'): os.mkdir('datasets_1')
            dataset_task_path = os.path.join('datasets_1', dataset_id)
            with open(dataset_task_path, 'wb') as f:
                pickle.dump(dataset_id, f)

            # construct deployment info
            datasets_info = { dataset_id: { 'name': dataset_name, 'project': dataset_proj} }

        # write to json
        add_to_json(deploy_info, 'datasets', datasets_info)

        return dataset_name, dest

def export_basedockers():
    # TODO: to export the base docker used
    pass

def export_preprocess():
    # TODO: to export the preprocess codes used
    pass

def get_args():
    parser = argparse.ArgumentParser(description='exporting')
    parser.add_argument(
        '--serving-service-id',
        default='cd423197c94344c4b1d41a5cd2408770',
        help='serving service id')
    return parser.parse_args()


if __name__ == "__main__":
    Task.init(
        project_name='Exporting2Standalone',
        task_name=str(datetime.now().date()),
        output_uri=True)
    args = get_args()
    export_from_development(args)
