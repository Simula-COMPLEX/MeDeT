"""
Created on September 05, 2023

@author: Hassan Sartaj
@version: 1.0
"""

import os
import torch
import learn2learn as l2l
from torch.utils.data import Dataset
from torch import nn, optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

file_separator = ";"


# -------------------
# Model Architecture
# -------------------
class DTModel(nn.Module):
    def __init__(self, infeatures=19, outfeatures=1, dim=32):
        super(DTModel, self).__init__()
        self.linear1 = nn.Linear(infeatures, dim).to(torch.float64)
        self.linear2 = nn.Linear(dim, outfeatures).to(torch.float64)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# ---------------------------------
# Meta Dataset & Taskset Creation
# ---------------------------------

feature_cols = None
label_outs = None
status_codes_map = {}


class DeviceDataset(Dataset):

    def __init__(self, file_name=None, device_name=None, i_features=None):
        df = pd.read_csv(file_name, sep=file_separator, skipinitialspace=True)
        df.iloc[:, 0:feature_cols - 1].astype(float)
        x = df.iloc[:, 0:feature_cols - 1].values
        y = df.iloc[:, feature_cols - 1].values

        # if feature_cols < i_features:
        #     diff = i_features - feature_cols
        #     df1 = pd.DataFrame(x)
        #     for d in range(diff + 1):
        #         df1["x" + str(d)] = 1
        #     x = df1.values

        self.x_train = torch.tensor(x, dtype=torch.float64)
        self.y_train = torch.tensor(y, dtype=torch.float64)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def create_data_task_sets(file_name, devices_names, ways=1, shots=1, num_tasks=1, i_features=1):
    for dev_n in devices_names:
        if dev_n in file_name:
            device_name = dev_n
            break
    deviceDs = DeviceDataset(file_name, device_name, i_features)
    dataset = l2l.data.MetaDataset(deviceDs)

    transforms = [  # Easy to define your own transform
        l2l.data.transforms.NWays(dataset, n=ways),
        l2l.data.transforms.KShots(dataset, k=2 * shots),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]

    # create taskset
    taskset = l2l.data.Taskset(dataset, transforms, num_tasks=num_tasks)

    # print("taskset: ")
    # for task in taskset.dataset:
    #     X, y = task
    #     print("X: ", X, " - y: ", y)
    #     break

    return taskset


# ---------------------------
# Meta-learning - Training
# --------------------------

def train_dtmodel(devices_names, devices_params, processed_datafiles, ml_root_dir, ex_name, shots, num_tasks, maml_lr, meta_lr, total_iterations, tps, adaption_steps):
    i_features = max(devices_params)

    # create initial tasksets for all files
    for datafile in processed_datafiles:
        taskset = create_data_task_sets(file_name=datafile, devices_names=devices_names, ways=1, shots=shots,
                                        num_tasks=num_tasks,
                                        i_features=i_features)

    scodes = []
    for dn in devices_names:
        scodes.append(len(status_codes_map[dn].items()))

    o_features = max(scodes)

    # iterate each data file
    for datafile in processed_datafiles:
        print("Loading data from ", datafile)
        # create_data_task_sets(file_name=datafile, ways=1, shots=shots, num_tasks=num_tasks)
        for dev_n in devices_names:
            if dev_n in datafile:
                d_classes = len(status_codes_map[dev_n])

        ways = d_classes

        taskset = create_data_task_sets(file_name=datafile, devices_names=devices_names, ways=ways, shots=shots, num_tasks=num_tasks,
                                        i_features=i_features)
        global feature_cols
        # create the model
        model = DTModel(infeatures=i_features, outfeatures=o_features, dim=128)
        maml = l2l.algorithms.MAML(model, lr=maml_lr)
        opt = optim.Adam(maml.parameters(), lr=meta_lr)
        loss_func = nn.CrossEntropyLoss()
        print("model: ", model)

        # Training
        device = torch.device("cpu")
        for iteration in range(total_iterations):
            iteration_error = 0.0
            iteration_acc = 0.0
            for _ in range(tps):
                learner = maml.clone()
                train_task = taskset.sample()
                data, labels = train_task
                data = data.to(device)
                labels = labels.to(device)

                # Separate data into adaptation/evaluation sets
                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[np.arange(shots * ways) * 2] = True
                evaluation_indices = torch.from_numpy(~adaptation_indices)
                adaptation_indices = torch.from_numpy(adaptation_indices)
                adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
                evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

                # print("adaptation_data: ", adaptation_data, " \n - label: ", adaptation_labels)
                # print("evaluation_data: ", evaluation_data, " \n - label: ", evaluation_labels)

                # Fast Adaptation
                for step in range(adaption_steps):
                    output = learner(adaptation_data)
                    train_error = loss_func(output, adaptation_labels)
                    learner.adapt(train_error)

                # Compute validation loss
                predictions = learner(evaluation_data)

                valid_error = loss_func(predictions, evaluation_labels)
                valid_error /= len(evaluation_data)
                max_pred = predictions.softmax(dim=1)
                _, max_indices = torch.max(max_pred, 1)
                valid_accuracy = accuracy_score(evaluation_labels.detach().numpy(), max_indices.detach().numpy())
                iteration_error += valid_error
                iteration_acc += valid_accuracy

            iteration_error /= tps
            iteration_acc /= tps

            if iteration % 50 == 0:
                print("Iteration # {} - Loss : {:.3f} Acc : {:.3f}".format(iteration, iteration_error.item(),
                                                                           iteration_acc))

            # Take the meta-learning step
            opt.zero_grad()
            iteration_error.backward()
            opt.step()

        print("Iteration # {} - Loss : {:.3f} Acc : {:.3f}".format(iteration, iteration_error.item(), iteration_acc))

        print("Done! Saving iDT model...")
        model_file = datafile.split("/")[len(datafile.split("/")) - 1].replace("prodata.csv",
                                                                               "idtmodel")  # .replace("prodata.csv", "dtmodel").replace("mldt-ds/", "")
        torch.save(model, ml_root_dir + ex_name + model_file + ".pth")


# -----------------------------
# Meta-learning - Fine-tuning
# -----------------------------
def finetune_dtmodel(devices_names, devices_params, processed_datafiles, ml_root_dir, ex_name, shots, num_tasks, maml_lr, meta_lr, total_iterations, tps, adaption_steps):
    i_features = max(devices_params)
    new_ml_root = ml_root_dir + ex_name
    for idt_file in os.listdir(new_ml_root):
        if ".pth" not in idt_file:
            continue
        dev_name = idt_file.split("_")[0]
        print("Loading model for ", dev_name)
        idt_model = torch.load(new_ml_root + idt_file)
        for datafile in processed_datafiles:
            if dev_name in datafile:
                continue

            print("Loading data from ", datafile)
            for dev_n in devices_names:
                if dev_n in datafile:
                    d_classes = len(status_codes_map[dev_n])

            ways = d_classes

            taskset = create_data_task_sets(file_name=datafile, ways=ways, shots=shots, num_tasks=num_tasks,
                                            i_features=i_features)
            global feature_cols
            # use trained the model
            # model = DTModel(infeatures=feature_cols-1, outfeatures=d_classes, dim=128) #dim=shots*ways*2
            # with torch.no_grad():
            #   idt_items = list(idt_model.state_dict().items())
            #   my_model_kvpair=model.state_dict()
            #   count=0
            #   for key,value in my_model_kvpair.items():
            #     layer_name,weights=idt_items[count]
            #     # my_model_kvpair[key]=weights
            #     my_model_kvpair[key] = weights.detach().clone()
            #     count+=1
            #   model.load_state_dict(my_model_kvpair)
            #   print("model: ", model)

            # print("model.weight: ", idt_model.state_dict())
            # model.load_state_dict(idt_model.state_dict(), strict=False)
            # model2 = copy.deepcopy(idt_model)
            # model.load_state_dict(model2.state_dict())

            maml = l2l.algorithms.MAML(idt_model, lr=maml_lr)
            opt = optim.Adam(maml.parameters(), lr=meta_lr)
            loss_func = nn.CrossEntropyLoss()
            print("model: ", idt_model)

            device = torch.device("cpu")
            for iteration in range(total_iterations):
                iteration_error = 0.0
                iteration_acc = 0.0
                for _ in range(tps):
                    learner = maml.clone()
                    train_task = taskset.sample()
                    data, labels = train_task
                    data = data.to(device)
                    labels = labels.to(device)

                    # Separate data into adaptation/evaluation sets
                    adaptation_indices = np.zeros(data.size(0), dtype=bool)
                    adaptation_indices[np.arange(shots * ways) * 2] = True
                    evaluation_indices = torch.from_numpy(~adaptation_indices)
                    adaptation_indices = torch.from_numpy(adaptation_indices)
                    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
                    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

                    # print("b adaptation_data: ", adaptation_data, " \n - label: ", adaptation_labels)
                    # print("b evaluation_data: ", evaluation_data, " \n - label: ", evaluation_labels)

                    # Fast Adaptation
                    for step in range(adaption_steps):
                        output = learner(adaptation_data)
                        # output=torch.max(output.data,1)
                        # print("output", output)
                        # output = output.reshape(shots*2)
                        # output = output.to(torch.float64)

                        train_error = loss_func(output, adaptation_labels)
                        # train_error = train_error.to(torch.float32)
                        # print(train_error)
                        learner.adapt(train_error)

                    # print("a adaptation_data: ", adaptation_data, " \n - label: ", adaptation_labels)
                    # print("a evaluation_data: ", evaluation_data, " \n - label: ", evaluation_labels)

                    # Compute validation loss
                    predictions = learner(evaluation_data)
                    # predictions = predictions.reshape(shots*2)
                    # print("predictions: ", predictions)
                    # print("evaluation_labels: ", evaluation_labels)

                    valid_error = loss_func(predictions, evaluation_labels)
                    # print("valid_error: ", valid_error)
                    valid_error /= len(evaluation_data)
                    # valid_accuracy = accuracy(predictions, evaluation_labels)
                    # print("predictions: ", predictions)
                    max_pred = predictions.softmax(dim=1)
                    _, max_indices = torch.max(max_pred, 1)
                    # print("max_indices: ", max_indices)
                    valid_accuracy = accuracy_score(evaluation_labels.detach().numpy(), max_indices.detach().numpy())
                    # valid_accuracy = balanced_accuracy_score(evaluation_labels.detach().numpy(),max_indices.detach().numpy())
                    iteration_error += valid_error
                    iteration_acc += valid_accuracy

                iteration_error /= tps
                iteration_acc /= tps

                if iteration % 10 == 0:
                    print("Iteration # {} - Loss : {:.3f} Acc : {:.3f}".format(iteration, iteration_error.item(),
                                                                               iteration_acc))

                # Take the meta-learning step
                opt.zero_grad()
                # print("iteration_error: ", iteration_error)
                # iteration_error = iteration_error.to(torch.float64)
                iteration_error.backward()
                opt.step()

            model_file = datafile.split("/")[len(datafile.split("/")) - 1].replace("prodata.csv",
                                                                                   "adtmodel-" + dev_name[
                                                                                       0])
            print("Done! Saving aDT model: ", model_file)
            torch.save(idt_model, new_ml_root + model_file + ".pth")
