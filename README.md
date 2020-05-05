# TrackerCSK ![](https://img.shields.io/badge/license-MIT-blue)
A Pure CSK implementation in python, adapted from the python version [circulant_matrix_tracker](https://github.com/rodrigob/circulant_matrix_tracker)

## Reference Details regarding the tracking algorithm can be found in the following paper:

[Exploiting the circulantstructure of tracking-by-detection with kernels](https://dl.acm.org/doi/10.1007/978-3-642-33765-9_50).    
Henriques J F, Caseiro R, Martins P, et al.  
European conference on computer vision (ECCV), 2012 IEEE Conference.

## Usage
The implementation uses got10k for tracking performance evaluation. which ([GOT-10k toolkit](https://github.com/got-10k/toolkit)) is a visual tracking toolkit for VOT evaluation on main tracking datasets.
* Run test.py to evaluate on OTB or/and VOT dataset.
```cmd 
>> python test.py 
```
