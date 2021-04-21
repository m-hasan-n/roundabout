# Maneuver-based Anchor Trajectory Hypotheses
We extend existing recurrent encoder-decoder models to be advantageously
combined with anchor trajectories to predict vehicle behaviors
on a roundabout. Drivers’ intentions are encoded by a set
of maneuvers that correspond to semantic driving concepts.
Accordingly, our model employs a set of maneuver-specific
anchor trajectories that cover the space of possible outcomes
at the roundabout. The proposed model can output a multi-
modal distribution over the predicted future trajectories based
on the maneuver-specific anchors. We evaluate our model using
the public RounD dataset.

#
![model image](roundabout_model_2.png "Model overview")

## RounD Dataset Pre-processing
The [RounD Dataset](https://www.round-dataset.com/#download) is a new dataset of naturalistic
road user trajectories recorded at German roundabouts. 
Download the dataset to the 'rounD' directory, then run the following MATLAB script:
```
preprocess_rounD.m
```
This will do the required pre-processing, split the dataset into train, validation and test subsets, and save such subsets into the 'data' directory.

## Model Arguments
The default network arguments are in:
```
model_args.py 
```
You can set the required experiment arguments in this script. For example: 
* args['ip_dim'] selects the input dimensionality (2D or 3D).

* args['use_intention'] and args['use_anchors'] are Boolean variables that choose whether using intention prediction and anchor trajectories or not.

## Model Training and Evaluation
The model structure is coded in 'model.py'. After setting the required experiment arguments, create a 'trained_models' directory to save the trained models.
You can start model training by running:
```
train.py
```

To test a trained model, first create an 'eval_res' directory, then run:
```
evaluate.py
```
which will load and test the trained model defined by the selected model arguments. The RMSE results will be saved as csv files to the 'eval_res' directory. 

## Citation
If you find this code useful for your research, please cite [our work](http://arxiv.org/):

* Mohamed Hasan, Evangelos Paschalidis, Albert Solernou, He Wang, Gustav Markkula and Richard Romano, "Maneuver-based Anchor Trajectory Hypotheses at Roundabouts", under review IROS 2021.

## License
This project is licensed under the MIT License - see the 
[LICENSE.md](LICENSE.md) file for details.

