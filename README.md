# deploy2standalone

# Deploy Info Structure
- This is the info that will be exported by export.py and used by import.py to bring the deployment to standalone environment.

```
{
  developer: 'Kai Yong',
  exported_on: '2022-12-21 11:15:30.067380',
  
  # assuming deployment to standalone is based on what was pushed to Serving Service in Development Env
  serving_id: < id of serving service from clearml (dev)>,
  
  # model deployed to the endpoint - to check this again
  model: {
    id: <id of the model from clearml (dev)>,
    # e.g. http://localhost:8081/<proj>/<task name>.<task id>>/models/<model name>.pt
    url: <url of the model uploaded to clearml (dev)>,    
  },
  
  # main deploy info required for importing script to carry out importing
  tasks: [
    {
      id: <id of experiment task from clearml (dev)>,
      # models generated by the experiment task
      models: [
        {
          name: <name of model>,
          model: <id of model>
        }
      ],
      # datasets used in this experiment task
      [<id of dataset>, ...]
    },
    <next task> ....
  ],
  
  # datasets details
  datasets: {
    <id of dataset>: {
      name: <name of dataset in clearml (dev)>,
      project: <project which dataset is organised in clearml (dev)>,      
    },
    <next dataset>, ....
  }
}
```

# Tasks
- [ ] Draft a rough concept. Check back with team.
   - add a diagram to map the dev and standalone setup.
- [x] Enable export and import of tasks.
- [x] Enable export and import of models.
- [x] Enable export and import of datasets.
- [ ] Enable export and import of base dockers.
- [ ] Enable export and import of preprocess codes.
- [ ] Enable export and import of training codes
- [x] Enable generation of deploy info and import info.
- [ ] Generalise the scripts with TF framework.
- [ ] Create a docker image to do the exporting and importing. Add the scripts as entrypoint. 
  - Setup the AIP equivalent and test. 
- [ ] Test it with an offline environment.
- [ ] Improve deploy info and import info.
- [ ] Test it with own Triton image, instead of clearml-serving-triton.
- [ ] Test with more complex example (YOLOv5).
- [ ] Test with more complex example (a customised project).
- [ ] Test with retraining pipelines to find out additional exporting / importing required.
- [ ] Run pylint. 
- [ ] Draft out how to sync standalone env back to dev, e.g. extra datasets annotated and used for retraining. newer retrained model in standalone env.
  - list down possible cases and see how to sync back. 
  - see if can "auto-sync" back. 
