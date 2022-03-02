import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np

import os
import time
from datetime import datetime

from tqdm import tqdm

import sys
from loguru import logger

import matplotlib.pyplot as plt
from datetime import datetime

def test():
    print("model creation and training")
    
class augmented_data(Dataset):
    """
    Augmented dataset

    This class inherits from PyTorch's Dataset class, which allows for overloading the initializer and getter to contain a transform operation.
    """
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        return item
    
# Function used to choose computation device.
def set_device(var, use_cuda: bool):
    if use_cuda == True:
        if torch.cuda.is_available():
            var = var.cuda()
        else:
            var = var.cpu()
    else:
        var = var.cpu()

    return var

class Net(object):
    def __init__(
        self,
        pretrained_model,
        num_classes: int,
        mode: str,
        ckpt_dir: str,
        cuda: bool = False,
        labels: list = [],
        lr: float = 0.0):
        
        self.mode = mode
        self.model = pretrained_model
        self.num_classes = num_classes
        self.lr = lr
        self.cuda = cuda
        self.ckpt_dir = ckpt_dir
        self.labels = labels
        
        self.epoch_list = []
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        
        # Check whether the mode is correct
        assert self.mode == "feature_extractor" or self.mode == "feature_composer"

        # Check if checkpoint directory exists
        assert os.path.exists(ckpt_dir)

        # Prepare model for computation on the selected device
        self.model = set_device(self.model, self.cuda)

        # Extract the input size based on the second to last layer
        try:
            self.input_size = self.model.classifier[-1].in_features
        except:
            self.input_size = self.model.fc.in_features

        # Introduce a new layer of computation
        self.classification_layer = nn.Linear(
            in_features=self.input_size, 
            out_features=self.num_classes
        )
        self.softmax_activation = nn.Softmax(dim=1)

        # Set the weights and biases accordingly
        with torch.no_grad():
            self.classification_layer.weight = torch.nn.Parameter(set_device(torch.randn((self.num_classes, self.input_size)) * 1e-5, self.cuda))
            self.classification_layer.bias = torch.nn.Parameter(set_device(torch.randn(self.num_classes) * 1e-5 + 1, self.cuda))

        # Prepare the layer for computation on the selected device
        self.classification_layer = set_device(nn.Sequential(
            self.classification_layer, 
            self.softmax_activation
        ), self.cuda)

        # Replace the pretrained classification layer with the custom classification layer
        try:
            self.model.classifier[-1] = self.classification_layer
            
            # Total number of pretrained layers 
            # (except last classification layer)
            self.num_pretrained_layers = len(list(self.model.modules())) - len(list(self.model.classifier))
        except:
            self.model.fc = self.classification_layer
            
            # Total number of pretrained layers 
            # (except last fully connected layer)
            self.num_pretrained_layers = len(list(self.model.modules())) - len(list(self.model.fc))

        # Training mode
        # Number of layers to activate and freeze
        self.hm_layers_to_activate, self.hm_layers_to_freeze = 0, 0

        # The choice will only be given if it is 
        # the feature extractor that it is training.
        if self.mode == "feature_extractor":
            # User choice
            while(True): #while loop to ask user for training model
                print("""
                        Choose a mode in which you wish to train:\n
                        1) Shallow-tuning (Fast, but inaccurate)\n
                        2) Deep-tuning (Slow and requires a lot of data, but accurate)\n
                        3) Fine-tuning
                        """)

                while(True): #prompt user to enter a valid number
                    try: #if valid number is selected, it exits otherwise repeats
                        self.training_mode = int(input("> "))
                        break
                    except ValueError: #handle the exception if it is anything else than int
                        logger.warning("Please enter an integer")

                if(1 <= self.training_mode <= 3): #check if number is between 1 and 3, if so break 
                    break
            
            logger.info(f"training_mode {self.training_mode} selected") #add number to logging

            # If the user chose the fine-tuning method, 
            # prepare the layers for freezing and training respectively
            if self.training_mode == 3:
                print(f"Pretrained model has {self.num_pretrained_layers} layers.")
                
                # How many layers to activate 
                # (prepare their weights for gradient descent)
                self.hm_layers_to_activate = int(input("> How many layers to train?: "))
                while self.hm_layers_to_activate < 0 and self.hm_layers_to_activate > self.num_pretrained_layers:
                    self.hm_layers_to_activate = int(input("> How many layers to train?: "))
                # How many layers to freeze
                # (how many to omit when executing gradient descent)
                self.hm_layers_to_freeze = self.num_pretrained_layers - self.hm_layers_to_activate
        else:
            self.hm_layers_to_freeze = self.num_pretrained_layers

        # Set the save path, freeze or unfreeze the gradients based on the mode and define appropriate optimizers and schedulers.
        # Feature extractor => Freeze all gradients except the custom classification layer
        # Feature composer => Unfreeze / Activate all gradients
        now = datetime.now()
        now = f'{str(now).split(" ")[0]}_{str(now).split(" ")[1]}'.split(".")[0].replace(':', "-")
        if self.mode == "feature_extractor":
            self.save_name = f"DeTraC_feature_extractor_{now}.pth"

            if self.training_mode == 1:
                print("Freezing all pretrained layers. Activating only classification layer")
                for param in self.model.parameters():
                    param.requires_grad = False
                try:
                    for param in self.model.classifier[-1].parameters():
                        param.requires_grad = True
                except:
                    for param in self.model.fc.parameters():
                        param.requires_grad = True

            elif self.training_mode == 2:
                print("Activating all layers")
                for param in self.model.parameters():
                    param.requires_grad = True

            else:
                print(f"Freezing {self.hm_layers_to_freeze} layers and activating {self.hm_layers_to_activate}.")
                for i, param in enumerate(self.model.parameters()):
                    if i <= self.hm_layers_to_freeze:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                try:
                    for param in self.model.classifier[-1].parameters():
                        param.requires_grad = True
                except:
                    for param in self.model.fc.parameters():
                        param.requires_grad = True                

            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                nesterov=False,
                weight_decay=1e-3
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.9,
                patience=3
            )

        else:
            self.save_name = f"DeTraC_feature_composer_{now}.pth"

            for param in self.model.parameters():
                param.requires_grad = True

            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=0.95,
                nesterov=False,
                weight_decay=1e-4
            )

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=0.95,
                patience=5
            )

        self.ckpt_path = os.path.join(self.ckpt_dir, self.save_name)

        # Define the loss.
        # Categorical crossentropy is a negative log likelihood loss, where the logit is the log of the model's output, and the label is the argmax from the list of labels
        self.criterion = nn.NLLLoss()
        
    def train_step(self,  train_loader: DataLoader):
        # Set model in train mode
        self.model.train()
        # Initialize running error and running accuracy metrics
        running_error = 0.0
        running_correct = 0

        # Iterate through the data
        for features, labels in train_loader:
            if self.mode == "feature_extractor":
                # Here we permute because pretrained models in Pytorch require inputs of shape (batch_size, num_channels, width, height)
                features = features.permute(0, 3, 1, 2)

            # Load data to the desired computation device
            features = set_device(features, self.cuda)
            labels = set_device(labels, self.cuda)

            # Reset the optimizer's gradients
            self.optimizer.zero_grad()
            # Enable gradients for gradient descent
            with torch.set_grad_enabled(True):
                # Prediction
                pred = self.model(features)

                # Cross entropy loss
                loss = self.criterion(torch.log(pred), torch.max(labels, 1)[1])

                # Backprop
                loss.backward()

                # Optimizer step
                self.optimizer.step()

            # Labels for metrics
            _, pred_idx = torch.max(pred.data, 1)
            _, true_idx = torch.max(labels.data, 1)
            # Metrics calculations
            running_error += loss.item() * features.size(0)
            running_correct += (pred_idx == true_idx).float().sum()

        # Total error
        err = running_error / len(train_loader.dataset)
        
        # Total accuracy
        acc = 100 * running_correct / len(train_loader.dataset)
        return err, acc
    
    def save(
        self, 
        epoch: int, 
        train_loss: float, 
        train_acc: float, 
        val_loss: float, 
        val_acc: float
    ):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "labels": self.labels,
            "num_classes": self.num_classes
        }, self.ckpt_path)   
         
    def infer_using_pretrained_layers_without_last(self, features: np.ndarray):
        # Keep track of the last layer
        last_layer_to_restore = list(self.model.children())[-1]
        last_layer = list(self.model.children())[-1][:-1]

        # Remove last layer
        while type(last_layer[-1]) != nn.modules.linear.Linear:
            last_layer = last_layer[:-1]

        # Update the model architecture
        try:
            self.model.classifier = last_layer
        except:
            self.model.fc = last_layer

        # We'll use the CPU for computation here, as continuous GPU inference tends to give an "Out of memory" error.
        self.model = self.model.cpu()

        # Convert to tensor
        features = torch.Tensor(features)

        # Disable gradients
        with torch.no_grad():
            # Predict
            features = features.permute(0, 3, 1, 2)
            pred = self.model(features)
        
        # Convert to array
        pred = pred.numpy()

        # Restore layer
        try:
            self.model.classifier = last_layer_to_restore
        except:
            self.model.fc = last_layer_to_restore

        # Return prediction
        return pred

    def validation_step(self, validation_loader: DataLoader):
        # Set model in evaluation mode
        self.model.eval()

        # Initialize running error and running accuracy metrics
        running_error = 0.0
        running_correct = 0

        # Iterate through the data
        for features, labels in validation_loader:
            if self.mode == "feature_extractor":
                features = features.permute(0, 3, 1, 2)

            # Load data to the desired computation device
            features = set_device(features, self.cuda)
            labels = set_device(labels, self.cuda)

            # Reset the optimizer's gradients
            self.optimizer.zero_grad()

            # Disable gradients as there is no training done in the validation step
            with torch.no_grad():
                # Prediction
                pred = self.model(features)

                # Cross-entropy loss
                loss = self.criterion(torch.log(pred), torch.max(labels, 1)[1])

            # Labels for metrics
            _, pred_idx = torch.max(pred.data, 1)
            _, true_idx = torch.max(labels.data, 1)

            # Metrics calculations
            running_error += loss.item() * features.size(0)
            running_correct += (pred_idx == true_idx).float().sum().item()

        # Total error
        err = running_error / len(validation_loader.dataset)
        
        # Total accuracy
        acc = 100 * running_correct / len(validation_loader.dataset)

        return err, acc
    
    def infer(
        self, 
        input_data: np.ndarray, 
        ckpt_path: bool = None, 
        use_labels: bool = False
    ):
        # Disable gradients. We're not training.
        with torch.no_grad():
            # Convert the input data into a tensor if needed
            if type(input_data) != torch.Tensor:
                input_data = torch.Tensor(input_data)

            # Reshape if appropriately.
            input_data = input_data.reshape(-1, 3, 224, 224)

            # Prediction
            output = self.model.cpu()(input_data).numpy()

            if use_labels == True:

                # Check if a model path is given
                assert ckpt_path != None

                # Create a dictionary of the shape: <label> : <confidence>
                labeled_output = {}
                labels = self.load_labels_for_inference(ckpt_path)
                for label, out in zip(labels, output[0]):
                    labeled_output[label] = out
                return labeled_output
            else:
                return output
            
    def fitv2(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        resume: bool,
        accuracy_folder_path: str,
        loss_folder_path: str,
        feature_type_acc: str,
        feature_type_loss: str):
        
        train_loader = None
        
        if self.cuda == True:
            torch.backends.cudnn.benchmark = True

        # Mode choice
        if self.mode == "feature_extractor":
            train_loader = DataLoader(dataset=list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
            validation_loader = DataLoader(dataset=list(zip(x_test, y_test)), shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
            #train_loader = DataLoader(dataset=list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
            #validation_loader = DataLoader(dataset=list(zip(x_test, y_test)), shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)
        else:
            # We want to augment the data when in feature composition mode.
            train_transform = transforms.Compose([
                transforms.ToPILImage("RGB"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
            ])

            val_transform = transforms.Compose([
                transforms.ToPILImage("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
            ])

            x_train_ds = augmented_data(x_train, train_transform)
            x_test_ds = augmented_data(x_test, val_transform)

            train_loader = DataLoader(
                dataset=list(zip(x_train_ds, y_train)), 
                shuffle=True, 
                batch_size=batch_size
            )
            validation_loader = DataLoader(
                dataset=list(zip(x_test_ds, y_test)), 
                shuffle=True, 
                batch_size=batch_size
            )

        start_epoch = 0
        # If the user chooses to resume
        if resume == True:

            # List of checkpoints
            ckpt_paths_list = []
            for i, ckpt_path in enumerate(os.listdir(self.ckpt_dir)):
                if self.mode == "feature_extractor":
                    if "feature_extractor" in ckpt_path:
                        print(f"{i + 1}) {ckpt_path}")
                        ckpt_paths_list.append(ckpt_path)
                else:
                    if "feature_composer" in ckpt_path:
                        print(f"{i + 1}) {ckpt_path}")
                        ckpt_paths_list.append(ckpt_path)

            # Check if there are any available checkpoints
            assert len(ckpt_paths_list > 0)

            # Prompt user for a choice
            ckpt_path_choice = -1
            model_paths_list = "" #temp
            while ckpt_path_choice > len(model_paths_list) or ckpt_path_choice < 1:
                ckpt_path_choice = int(input(f"Which model would you like to load? [Number between 1 and {len(ckpt_paths_list)}]: "))

            ckpt_path = os.path.join(
                self.ckpt_dir, ckpt_paths_list[ckpt_path_choice - 1])

            # Load said checkpoint and it's elements.
            checkpoint = torch.load(self.ckpt_path)
            start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Training loop
            progress_bar = tqdm(range(start_epoch, epochs))
            progress_bar.set_description("Resuming training...")
            for epoch in progress_bar:
                train_loss, train_acc = self.train_step(train_loader)
                val_loss, val_acc = self.validation_step(validation_loader)

                # Regularization
                self.scheduler.step(val_loss)

                # Save condition
                if epochs >= 10:
                    if (epoch + 1) % (epochs // 10) == 0:
                        self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)
                else:
                    self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)

                progress_bar.set_description(
                    f"[Epoch {epoch + 1} stats]: train_loss = {train_loss} | train_acc = {train_acc}% | val_loss = {val_loss} | val_acc = {val_acc}%")
        
        else:
            progress_bar = tqdm(range(start_epoch, epochs))
            progress_bar.set_description("Starting training...")

            for epoch in progress_bar:
                train_loss, train_acc = self.train_step(train_loader)
                val_loss, val_acc = self.validation_step(validation_loader)
                
                # Regularization
                self.scheduler.step(val_loss)

                # Save condition
                if epochs >= 10:
                    if (epoch + 1) % (epochs // 10) == 0:
                        self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)
                else:
                    self.save(epoch, train_loss,
                                  train_acc, val_loss, val_acc)
                #print("epoch: " + str(epoch) + "  train_loss: " + str(train_loss) + 
                # "  train_acc: " + str(train_acc) + "  val_loss: " + str(val_loss) + "  val_acc: " + str(val_acc))
                self.epoch_list.append(epoch+1)
                self.train_loss_list.append(train_loss)
                self.train_acc_list.append(train_acc)
                self.val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)
                progress_bar.set_description(
                    f"[Epoch {epoch + 1} stats]: train_loss = {train_loss:.2f} | train_acc = {train_acc:.2f}% | val_loss = {val_loss:.2f} | val_acc = {val_acc:.2f}%")

            timeNow = str(datetime.now())
            plt.figure()
            plt.plot(self.epoch_list, self.train_acc_list, 'bo', label='Training acc')
            plt.plot(self.epoch_list, self.val_acc_list, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.savefig(str(feature_type_acc) + "/" + str(timeNow + "_accuracy.png"))
            #plt.savefig(str(accuracy_folder_path) + "/" + str(timeNow + " " + feature_type_acc + " " +"accuracy.png").replace(" ","_"))
            #print(str(accuracy_folder_path) + "/" + str(timeNow + "_accuracy.png"))
            
            plt.figure()
            plt.plot(self.epoch_list, self.train_loss_list, 'bo', label='Training loss')
            plt.plot(self.epoch_list, self.val_loss_list, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.savefig(str(feature_type_loss) + "/" + str(timeNow + "_loss.png"))
            #plt.savefig("test2.png")
            #print(str(feature_type_loss) + "/" + str(timeNow + "_loss.png"))
            #plt.savefig(str(loss_folder_path) + "/" + str(timeNow + " " + feature_type_loss + " " +"loss.png").replace(" ","_"))
            
