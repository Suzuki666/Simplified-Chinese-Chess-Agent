Author: Zhongke Sun
Date: 25 March 2025

Environment is created in the 'environment' folder. Subfolder 'light_env' is used in the training process to accelerate the efficiency.

You will find 'Agent' - PolicyValueNetwork in the 'model' folder.

'tools' folder includes some components, it's necessary in the training process.

Please directly run 'train_loop.py' to run the entire training process.

The best model is stored in 'checkpoints'

Self play game results are stored in 'self play game results', you can check there.

If you want to evaluate the model, please run 'evaluate_model.py'

If you want to train a completely new model. Please delete the entire 'checkpoints' folder, the run 'init_model.py' to initialize an empty-weight model before 'train_loop.py'.

If you have any other question, please let me know via zhongke.sun@student-cs.fr