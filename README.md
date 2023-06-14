# Prediction of EEG signal using transformer model
Repository for the code from my bachelor's thesis <br>

src contains all code. Inside you find 
<ul>
  <li>main.py: used for trainig models</li>
  <li>trainBOHY.py: used for tuning models using Ray Tune</li>
  <li>test.py: a testing script</li>
  <li>data_utils.py: utility functions</li>
  <li>lightTransformer.py: code for the pytorch lightning implementation</li>
</ul>
model_checkpoints contains the trained models (except for transf3 as too big).
