# Maneuver-based Anchor Trajectory Hypotheses at Roundabouts
We extend existing recurrent encoder-decoder models to be advantageously
combined with anchor trajectories to predict vehicle behaviors
on a roundabout. Drivers’ intentions are encoded by a set
of maneuvers that correspond to semantic driving concepts.
Accordingly, our model employs a set of maneuver-specific
anchor trajectories that cover the space of possible outcomes
at the roundabout. The proposed model can output a multi-
modal distribution over the predicted future trajectories based
on the maneuver-specific anchors. We evaluate our model using
the public RounD dataset

![model image](roundabout_model_2.png "Model overview")