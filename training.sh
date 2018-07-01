IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

# The first retraining command iterates only 500 times. 
# You can very likely get improved results (i.e. higher accuracy) 
# by training for longer. To get this improvement, remove the parameter 
# --how_many_training_steps to use the default 4,000 iterations.

# Add this after bottlenecks to reduce training steps
# --how_many_training_steps=500 \

python3 -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos