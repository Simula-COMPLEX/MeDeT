# MeDeT: Few-Shot Meta-Learning for High-Fidelity Digital Twins of Medical Devices

An approach (MeDeT) for building, adapting, and operating digital twins (DTs) of medical devices to be used with healthcare IoT applications for testing.
MeDeT works in six phases: (i) _Data Generation_ - generates raw data for medical devices, (ii) _Data Preparation_ - preprocesses raw data for training, (iii) _Meta-learning_ - creates meta dataset & taskset, determines model architecture, and trains/fine-tunes with MAML algorithm, (iv) _Build DTs_ - creates model clones, storage, APIs, and JSON objects, (v) _DT Request Handler_ - processes requests from a healthcare IoT application during testing, and (vi) _DTs to Device
Communication_ - establishes DTs communication with medical devices. 

This work is a part of the Welfare Technology Solution project (WTT4Oslo).






[//]: # (The repository contains open-source implementation)


## Basic Requirements

* Machine: minimum 16GB RAM and 8-core processor 
* OS: MacOS or Windows 10
* IDE: PyCharm
* Python: 3.8 or higher 

## Dependencies

* PyTorch: 2.0.1
* learn2learn: 0.2.0
* scikit-learn: 1.3.0
* Pandas: 2.0.3
* Flask: 2.2.3
* Flask-RESTful: 0.3.9

