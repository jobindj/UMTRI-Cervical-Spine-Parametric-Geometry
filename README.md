# UMTRI Cervical Spine Parametric Geometry

Software for generating 2D and 3D cervical spine bone geometry as a function of anthropometry and posture.

Copyright 2017 University of Michigan

Author: mreed@umich.edu

Revision Date: 2017-02-25

Python version 3.x is required; tested with Python version 3.5.
The numpy library is required.

***********

```python
>>> from CSpine2DPCAR import *
    
>>> target_anthro = anthro(anthro.FEMALE, stature=1650, age=45, shs=0.52) # shs = sitting height / stature 
>>> pcar = PCARSpine2D()
>>> pcar.predict(target_anthro, delta_head_angle=-30) # delta head angle from neutral in degrees; negative is flexion
```

The `predict()` methods generates a spine using the PCAR model,
articulates the model according to the delta_head_angle, 
then writes the model to a file "SpineOut.tsv" that contains named points.

Opening SpineOut.tsv in Excel will automatically update the plot in ViewSpine.xlsx.

The `PCARSpine2D.predict()` method can also be called with keyword arguments:

```python
>>> pcar.predict(sex=anthro.FEMALE, stature=1750, age=45, delta_head_angle=20)
```

or with a list of anthro

```python
>>> pcar.predict(anthro=[1, 1750, 0.52, 45], delta_head_angle=20) # sex: -1=male, 1=male
```

The module can also be run from the command prompt with the anthro and posture as command-line arguments

```bash
$ python CSpine2DPCAR.py 1 1650 0.52 20 0
```
Note all five arguments (sex, stature, shs, age, and delta head angle) must be supplied. If none is supplied, a midsize female
spine is generated in the neutral posture.

A location parameter can be added to the predict() call to translate and rotate the model. 
location='C7' (or other level up to C2) will place the anterior-inferior margin of the body at the origin and align the inferior surface
of the body with the global x axis. Alternatively, enter a location and angle, e.g., [[40, 40], 30] will
translate the model by [40, 40] and rotate 30 degrees clockwise.

The segment positions and orientations are written at the end of the landmark file.

Running CSpine2DPCAR from the command line (executing the module) will automaticaly run CSpine3DFitting on the result.

***********

## CSpine3DFitting.py


Usage:

```python
>>> from CSpine3DFitting import *
>>> bm = BoneMapper()
```

The file SpineOut.tsv residing in the output directory is read. The 3D geometry in the data directory is mapped to the 2D geometry and output as OBJ and landmark files. The OBJ files can be read in meshlab and many other graphics packages.

Alternatively, from the command line

```
python CSpine3DFitting.py
```

The output directory is expected to contain a file called SpineOut.tsv containing the 2D landmarks. 
An alternative file can be supplied on the command line:

```
python CSpine3DFitting.py AlternativeSpine.tsv
```

Note that the path for the alternative file is relative to the module.
The data directory containing the bone mesh and landmark files must be in the same directory as the module.



## Reference:

Reed, M.P. and Jones, M.H. (2017). A Parametric Model of Cervical Spine Geometry and Posture. Technical Report. University of Michigan Transportation Research Intitute, Ann Arbor, MI