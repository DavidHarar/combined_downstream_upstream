# Summary
This folder covers the upstream-downstream integration. It is leaner than the two other parts of the project - the upstream_seq2seq and downstream classification task. The reason for that is, that it draws models, utils and data-loaders from each of them when needed.  
This folder is devoted to experiments and recording of integrations between the two tasks.



# Task 1 - See if imputation off-the-shelf helps
In this task, we check if running the downstream task on imputed signals is helpful. For the sake of this task we don't retrain the upstream seq2seq model. We do, however, retrain the exact same model as in the downstream.  
The things needed in order to make this test succeed are:  
    1. Dataloader that loads SHL observations and pass them through a model (the model should be loaded internaly by the dataloader, given a path).
    2. Inception model from downstream.

# Task 2 - Train the two networks together
In this task, we combine the two models into one big model, that will take partial signals, impute and classify together. This would allow the task of imputation to be tailored suit for SHL data. In order to fulfil this taks we need:
    1. Same DataLoader as in the downstream task
    2. Model generator, that takes two paths for downstream and upstream, and returns a combined model.



