import argparse
import os, pickle

# default pytorch import
import torch

# import lightning library
import pytorch_lightning as pl

# import trainer class, which orchestrates our model training
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# import our model class, to be trained
from MNISTClassifier import MNISTClassifier

# This is the main method, to be run when train.py is invoked
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=0) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('-m','--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('-tr','--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('-te','--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    #print(args)

    #tb_logger = TensorBoardLogger(save_dir="tensorboard", name="my_model")
    tb_logger = TensorBoardLogger(save_dir=args.output_data_dir+"/tensorboard",name="mnist_model")

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    mnistTrainer=pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, default_root_dir=args.output_data_dir, logger=tb_logger)

    # Set up our classifier class, passing params to the constructor
    model = MNISTClassifier(
        batch_size=args.batch_size, 
        train_data_dir=args.train, 
        test_data_dir=args.test
        )
    
    # Runs model training 
    mnistTrainer.fit(model)

    # Tests the model
    mnistTrainer.test(model)

    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    # Save the parameters used to construct the model - if you want to save model parameters
    """ print("Saving the model parameters")
    model_info_path = os.path.join(args.output_data_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'vocab_size_enc': 1,
            'vocab_size_dec': 2,
            'sos_token_input': 3,
        }
        pickle.dump(model_info, f)     """
    