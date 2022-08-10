#!/bin/bash

#evaluate baseline
python model_bilstm.py datasets/___0.01 datasets/___1.0  0.05 32 gpu
python model_bilstm.py datasets/___0.1  datasets/___1.0  0.05 32 gpu
python model_bilstm.py datasets/___0.3  datasets/___1.0  0.05 32 gpu
python model_bilstm.py datasets/___0.5  datasets/___1.0  0.05 32 gpu
python model_bilstm.py datasets/___1.0  datasets/___1.0  0.05 32 cpu

#evaluate MR replacement effect
python model_bilstm.py datasets/cMR0.01 datasets/cMR1.0  0.05 32 gpu
python model_bilstm.py datasets/cMR0.1  datasets/cMR1.0  0.05 32 gpu
python model_bilstm.py datasets/cMR0.3  datasets/cMR1.0  0.05 32 gpu
python model_bilstm.py datasets/cMR0.5  datasets/cMR1.0  0.05 32 gpu
python model_bilstm.py datasets/cMR1.0  datasets/cMR1.0  0.05 32 cpu

#evaluate BT replacement effect
python model_bilstm.py datasets/cBT0.01 datasets/cBT0.01 0.05 32 gpu
python model_bilstm.py datasets/cBT0.1  datasets/cBT0.1  0.05 32 gpu
python model_bilstm.py datasets/cBT0.3  datasets/cBT0.3  0.05 32 gpu
python model_bilstm.py datasets/cBT0.5  datasets/cBT0.5  0.05 32 gpu
python model_bilstm.py datasets/cBT1.0  datasets/cBT1.0  0.05 32 cpu

#evaluate SR0.2THE replacement effect
python model_bilstm.py datasets/cSR0.01f0.2pthe datasets/cSR0.01f0.2pthe 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.2pthe  datasets/cSR0.1f0.2pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.2pthe  datasets/cSR0.3f0.2pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.2pthe  datasets/cSR0.5f0.2pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.2pthe  datasets/cSR1.0f0.2pthe  0.05 32 cpu

#evaluate SR0.4THE replacement effect
python model_bilstm.py datasets/cSR0.01f0.4pthe datasets/cSR0.01f0.4pthe 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.4pthe  datasets/cSR0.1f0.4pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.4pthe  datasets/cSR0.3f0.4pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.4pthe  datasets/cSR0.5f0.4pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.4pthe  datasets/cSR1.0f0.4pthe  0.05 32 cpu

#evaluate SR0.6THE replacement effect
python model_bilstm.py datasets/cSR0.01f0.6pthe datasets/cSR0.01f0.6pthe 0.05 32 gpu 
python model_bilstm.py datasets/cSR0.1f0.6pthe  datasets/cSR0.1f0.6pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.6pthe  datasets/cSR0.3f0.6pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.6pthe  datasets/cSR0.5f0.6pthe  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.6pthe  datasets/cSR1.0f0.6pthe  0.05 32 cpu

#evaluate SR0.2FTX replacement effect
python model_bilstm.py datasets/cSR0.01f0.2pftx datasets/cSR0.01f0.2pftx 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.2pftx  datasets/cSR0.1f0.2pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.2pftx  datasets/cSR0.3f0.2pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.2pftx  datasets/cSR0.5f0.2pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.2pftx  datasets/cSR1.0f0.2pftx  0.05 32 cpu

#evaluate SR0.4FTX replacement effect
python model_bilstm.py datasets/cSR0.01f0.4pftx datasets/cSR0.01f0.4pftx 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.4pftx  datasets/cSR0.1f0.4pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.4pftx  datasets/cSR0.3f0.4pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.4pftx  datasets/cSR0.5f0.4pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.4pftx  datasets/cSR1.0f0.4pftx  0.05 32 cpu

#evaluate SR0.6FTX replacement effect
python model_bilstm.py datasets/cSR0.01f0.6pftx datasets/cSR0.01f0.6pftx 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.6pftx  datasets/cSR0.1f0.6pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.6pftx  datasets/cSR0.3f0.6pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.6pftx  datasets/cSR0.5f0.6pftx  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.6pftx  datasets/cSR1.0f0.6pftx  0.05 32 cpu

#evaluate SR0.2CLM replacement effect
python model_bilstm.py datasets/cSR0.01f0.2pclm datasets/cSR0.01f0.2pclm 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.2pclm  datasets/cSR0.1f0.2pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.2pclm  datasets/cSR0.3f0.2pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.2pclm  datasets/cSR0.5f0.2pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.2pclm  datasets/cSR1.0f0.2pclm  0.05 32 cpu

#evaluate SR0.4CLM replacement effect
python model_bilstm.py datasets/cSR0.01f0.4pclm datasets/cSR0.01f0.4pclm 0.05 32 gpu 
python model_bilstm.py datasets/cSR0.1f0.4pclm  datasets/cSR0.1f0.4pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.4pclm  datasets/cSR0.3f0.4pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.4pclm  datasets/cSR0.5f0.4pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.4pclm  datasets/cSR1.0f0.4pclm  0.05 32 cpu

#evaluate SR0.6CLM replacement effec
python model_bilstm.py datasets/cSR0.01f0.6pclm datasets/cSR0.01f0.6pclm 0.05 32 gpu
python model_bilstm.py datasets/cSR0.1f0.6pclm  datasets/cSR0.1f0.6pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.3f0.6pclm  datasets/cSR0.3f0.6pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR0.5f0.6pclm  datasets/cSR0.5f0.6pclm  0.05 32 gpu
python model_bilstm.py datasets/cSR1.0f0.6pclm  datasets/cSR1.0f0.6pclm  0.05 32 cpu