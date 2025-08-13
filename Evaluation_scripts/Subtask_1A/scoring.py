import json
import os
from sklearn.metrics import f1_score
import pandas as pd
import re

ROOT_DIR = '/app/'

### Set to False if you used inclusive indices in your solution
EXCLUSIVE_INDEX = True

''' 
1- Place the test XML file and a tsv of the expected solutions in the {reference_dir}
2- Place your prediction tsv in {prediction_dir}
'''

reference_dir = os.path.join(ROOT_DIR, 'input/', 'ref')
prediction_dir = os.path.join(ROOT_DIR, 'input/', 'res')
score_dir = os.path.join(ROOT_DIR, 'output/')

print('Reading reference')

try:
    test_xml = os.path.join(reference_dir, [xml_file for xml_file in os.listdir(reference_dir) if xml_file.endswith('.xml')][0])
except IndexError:
    raise FileNotFoundError('No XML file found in the reference directory. Contact the organizers if you see this error.')

if not os.path.exists(test_xml):
    raise FileNotFoundError(f'Reference XML file not found at {test_xml}. Ensure the file exists in the reference directory. Contact the organizers if you see this error.')

with open(test_xml, 'r', encoding='utf-8') as file:
    test_xml_content = file.read()

# Extract the question_id response mapping from the XML

all_questions = re.findall(r'<Question>(.*?)</Question>', test_xml_content, re.DOTALL)


qid_response_mapping = {}
for question in all_questions:
    question_id = re.search(r'<ID>(.*?)</ID>', question, re.DOTALL).group(1)
    response = re.search(r'<Response>(.*?)</Response>', question, re.DOTALL).group(1)

    qid_response_mapping[question_id] = response

print('Reading prediction')
try:
    pred_file = [file for file in os.listdir(prediction_dir) if file.endswith('.tsv')]
    
    if pred_file:
        pred_file = pred_file[0]

    else:
        pred_file = [file for file in os.listdir(prediction_dir) if not file.startswith('.')][0]   

    print(f'Found prediction file: {pred_file}')
except IndexError:
    raise FileNotFoundError('No prediction file found in the prediction directory. Ensure that there is exactly one file in TSV format.')

print('Contents of prediction directory (Should include your tsv submission):')
print(os.listdir(prediction_dir))


pred_data = pd.read_csv(os.path.join(prediction_dir, pred_file), sep='\t', header=None, names=['Question_ID', 'Span_Start', 'Span_End', 'Span_Type'], index_col=False)

try:
    ref_data = pd.read_csv(os.path.join(reference_dir, [hidden_file for hidden_file in os.listdir(reference_dir) if hidden_file.endswith('HIDDEN.tsv')][0]), sep='\t')
except IndexError:
    raise FileNotFoundError('No reference file found in the reference directory. Contact the organizers if you see this error.')

if any(elem not in ['Ayah', 'Hadith', 'No_Spans'] for elem in pred_data['Span_Type'].values):
    raise ValueError('Prediction file "Span_Type" column must contain only "Hadith", "Ayah", or "No_Spans".')

Normal_Text_Tag = 0
Ayah_Tag = 1
Hadith_Tag = 2

total_f1 = 0
count_valid_question = 0
for question_id in ref_data['Question_ID'].unique():

    if question_id not in pred_data['Question_ID'].values:
        print(f'Question ID {question_id} is missing from the prediction file.')
        continue
    

    count_valid_question += 1
    question_result = ref_data[ref_data['Question_ID'] == question_id]

    if question_result['Label'].values[0] == 'NoAnnotation':
        if pred_data[pred_data['Question_ID'] == question_id]['Span_Type'].values[0] == 'No_Spans':
            total_f1 += 1
        
    else:
        response_text = qid_response_mapping[question_id]

        pred_char_array = [Normal_Text_Tag for _ in range(len(response_text))]

        pred_result = pred_data[pred_data['Question_ID'] == question_id]

        if pred_result['Span_Type'].values[0] == 'No_Spans':
            # If the prediction is 'No_Spans', we skip the span marking
            continue

        for span_start, span_end, span_type in zip(pred_result['Span_Start'], pred_result['Span_End'], pred_result['Span_Type']):
            if not span_end < len(response_text):
                print('Submission is using exclusive indexing')
            
            assert span_start >= 0 and span_end <= len(response_text) and (span_end < len(response_text) or EXCLUSIVE_INDEX), f'Span start {span_start} or end {span_end} is out of bounds for response text length {len(response_text)}.'
            assert span_start <= span_end, f'Span start {span_start} must be less than span end {span_end}.'

            if not EXCLUSIVE_INDEX:
                span_end = span_end + 1

            if span_type == 'Ayah':
                pred_char_array[span_start:span_end] = [Ayah_Tag] * (span_end - span_start)
            elif span_type == 'Hadith':
                pred_char_array[span_start:span_end] = [Hadith_Tag] * (span_end - span_start)
        
        assert len(pred_char_array) == len(response_text), 'Length of predicted character array does not match the length of response text. Contact the organizers if you see this error.'

        truth_char_array = [Normal_Text_Tag for _ in range(len(response_text))]


        for span_start, span_end, span_type in zip(question_result['Span_Start'], question_result['Span_End'], question_result['Label']):
            
            ## Note that we do not modify span_end here as the solution tsv is exclusive of the end index
            if span_type == 'Ayah':
                truth_char_array[span_start:span_end] = [Ayah_Tag] * (span_end - span_start)
            elif span_type == 'Hadith':
                truth_char_array[span_start:span_end] = [Hadith_Tag] * (span_end - span_start)
        
        assert len(truth_char_array) == len(response_text), 'Length of truth character array does not match the length of response text.'

        f1 = f1_score(truth_char_array, pred_char_array, average='macro')
        total_f1 += f1

print('Calculating F1 Score')
f1_score_value = total_f1 / count_valid_question

print(f'F1 Score: {f1_score_value}')

scores = {
    'F1 Score': f1_score_value,
}

print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
