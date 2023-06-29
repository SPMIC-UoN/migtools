MIG tools for running CUDA applications
=======================================

Recent NVIDIA GPUs (e.g. A100) can be partitioned into multiple instances (MIGs) each
of which can be used as an independent GPU.

Unfortunately these MIG instances cannot be put into process-exclusive mode and the
default behaviour on running a CUDA application is to run on the first MIG partition
of the first GPU. This is clearly unacceptable for load balancing on an interactive
system.

These scripts are a hacky messy attempt to solve this problem.

Hopefully NVIDIA will enable process-exclusive mode for MIG partitions in a future
release or provide some other means of automatic load balancing.

getgpu
------
Running getgpu queries nvidia-smi to obtain a list of GPU MIG instances and any
processes that are *currently* running on them. It then returns the GUID of a random
MIG instance with no processes *currently* running. To run a CUDA application on
a 'free' MIG instance you would use:

    CUDA_VISIBLE_DEVICES=`getgpu` /bin/my_cuda_code

```getgpu``` can also be called with an integer index and will then return the GUID
of the corresponding MIG instance regardless of whether it appears to be in use or not.

There is no locking and no attempt to handle race conditions. Ideally applications should
use the script immediately before starting CUDA code to reduce the chance that another
application will be allocated the same MIG instance.

Optionally, a list of 'allowed' GUIDs can be put in /usr/local/share/gpus/ids - only
GUIDs from this list will be returned

To use the other tools, getgpu must be installed on the PATH, e.g. in /usr/local/bin

cudawrap
--------

This script can be used to wrap a CUDA executable. When an executable is linked to this 
script, the script will call an executable in the same directory but prefixed with
```CUDAWRAP_``` and use ```getgput``` to set ```CUDA_VISIBLE_DEVICES``` to the GUID
of a 'free' GPU MIG instance.

create_cudawrap
---------------

This script is used to set up a CUDA executable to be wrapped by ```cudawrap```. 

```create_cudawrap /path/to/my_cuda_executable```

will rename the original executable as ```/path/to/CUDAWRAP_my_cuda_executable```
and create a link ```/path/to/my_cuda_executable``` pointing at the ```cudawrap```
script.

This script assumes cudawrap has been installed in /usr/local/share/gpus

