import json
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import json

# Function to remove default diacritics from Arabic text
def remove_default_diac(s):
    out = s
    out = out.replace("َا", "ا")
    out = out.replace("ِي", "ي")
    out = out.replace("ُو", "و")
    out = out.replace("الْ", "ال")

    out = out.replace("ْ", "")

    out = out.replace("َّ", "َّ")
    out = out.replace("ِّ", "ِّ")
    out = out.replace("ُّ", "ُّ")
    out = out.replace("ًّ", "ًّ")
    out = out.replace("ٍّ", "ٍّ")
    out = out.replace("ٌّ", "ٌّ")

    out = out.replace("اَ", "ا")

    # التقاء الساكنين: ref:اخْتِتَامُ, sys:اِخْتِتامُ
    out = out.replace("اِ", "ا")
    out = out.replace("لِا", "لا")

    out = out.replace("اً", "ًا")

    return out

# Ensure the script is run in the correct directory
current_dir = os.path.dirname(os.path.abspath(__file__))

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'


quran_file = os.path.join(current_dir, 'quranic_verses.json')
if not os.path.exists(quran_file):
    raise FileNotFoundError(f'Quranic verses file not found at {quran_file}. Ensure the file exists in the current directory.')

print('Reading Quranic verses')

with open(quran_file, 'r', encoding='utf-8') as quran_file_handle:
    quranic_verses = json.load(quran_file_handle)

print('Contents of Quranic verses file:')
print(f'Number of Quranic verses loaded: {len(quranic_verses)}')

hadith_file = os.path.join(current_dir, 'six_hadith_books.json')
if not os.path.exists(hadith_file):
    raise FileNotFoundError(f'Hadith books file not found at {hadith_file}. Ensure the file exists in the current directory.')

print('Reading Hadith books')

with open(hadith_file, 'r', encoding='utf-8') as hadith_file_handle:
    hadith_books = json.load(hadith_file_handle)

print('Contents of Hadith books file:')
print(f'Number of Hadith books loaded: {len(hadith_books)}')

print('Reading prediction')
print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))

try:
    pred_file = [file for file in os.listdir(prediction_dir) if file.endswith('.tsv')][0]
    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')


pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', header=None, names=['Sequence_ID', 'Correction'], index_col=False)

try:
    ref_data = pd.read_csv(os.path.join(reference_dir, [hidden_file for hidden_file in os.listdir(reference_dir) if hidden_file.lower().endswith('hidden.tsv')][0]), sep='\t')
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')

print('Checking if predictions and references match in length')

num_correct = 0

hadith_db = [remove_default_diac(item['hadithTxt']) for item in hadith_books if item is not None and item['hadithTxt'] is not None] + [remove_default_diac(item['Matn']) for item in hadith_books if item is not None and item['Matn'] is not None]
quranic_db = [remove_default_diac(item['ayah_text']) for item in quranic_verses if item is not None and item['ayah_text'] is not None]

for seq_id in ref_data['Sequence_ID']:
    if seq_id not in pred_data['Sequence_ID'].values:
        raise ValueError(f'Sequence ID {seq_id} from reference file not found in prediction file.')
    
    pred_data_row = pred_data[pred_data['Sequence_ID'] == seq_id]

    if len(pred_data_row) != 1:
        raise ValueError(f'Multiple or no predictions found for Sequence ID {seq_id}. Ensure each Sequence ID has exactly one prediction.')
    
    pred_correction = pred_data_row['Correction'].values[0]
    # Remove default diacritics from the prediction
    pred_correction = remove_default_diac(pred_correction)

    ref_correction = ref_data[ref_data['Sequence_ID'] == seq_id]['Correction'].values[0]
    # Remove default diacritics from the reference
    ref_correction = remove_default_diac(ref_correction)

    if pred_correction == ref_correction or (ref_correction in pred_correction and (pred_correction in hadith_db or pred_correction in quranic_db)):
        num_correct += 1
    

print('Checking Accuracy')
accuracy = num_correct / len(ref_data)
print('Scores:')
scores = {
    'accuracy': accuracy,
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
