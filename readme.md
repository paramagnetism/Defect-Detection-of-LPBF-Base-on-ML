# Defect Detection of LPBF Base on ML
The code is use to generate 3D model from MPM and OT image from EOS LPBF 3D Printer.
To run the code, run in command:

```
pip install -r requirements.txt
```

## OT_mean.py /MPM_mean.py 
Used to calculate OT/MPM mean from image.
findrect.py is the base class for these two.

## MPMstdlzer.py
Generate standard MPM data from serval layers

## poly_regress.py 
Find the suitable function for melting pool depth prediction

## build3d.py 
Build the 3D model in folder Model3D/

## codes/rotater.py
The final model to predict only with OT image and initial setting STL file.

Other usages are all written in each file's title.