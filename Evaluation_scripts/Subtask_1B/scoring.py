import json
import os
from sklearn.metrics import accuracy_score
import pandas as pd

''' 
1- Place a tsv of the expected solutions in the {reference_dir}
2- Place your prediction tsv in {prediction_dir}
'''

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
try:
    pred_file = [file for file in os.listdir(prediction_dir) if file.endswith('.tsv')][0]
    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')

print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))

pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', header=None, names=['Sequence_ID', 'Label'])

try:
    ref_data = pd.read_csv(os.path.join(reference_dir, [hidden_file for hidden_file in os.listdir(reference_dir) if hidden_file.endswith('hidden.tsv')][0]), sep='\t')
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')

print('Checking if predictions and references match in length')

if len(pred_data) != len(ref_data):
    raise ValueError(f'Length of predictions ({len(pred_data)}) does not match length of references ({len(ref_data)}).')

if 'Label' not in pred_data.columns or 'Label' not in ref_data.columns:
    raise ValueError('Prediction file must contain a "Label" column.')

if any(elem not in ['Correct', 'Incorrect'] for elem in pred_data['Label'].values):
    raise ValueError('Prediction file "Label" column must contain only "Correct" or "Incorrect".')

prediction = pred_data['Label'].values
truth = ref_data['Label'].values

print('Checking Accuracy')
accuracy = accuracy_score(truth, prediction)
print('Scores:')
scores = {
    'accuracy': accuracy,
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
