from get_data import BasicDataProvider
import os

def sample_by_type(bdp, type_str, n=50):
    if type_str not in ['what', 'why', 'how', 'who', 'when', 'where', 'which']:
        raise AssertionError("Please provide only ['what', 'why', 'how', 'who', 'when', 'where', 'which'] for type_str")
    result = []
    while len(result) < n:
        sample = bdp.sampleImageQAPair()['qa_pair']
        if sample['type'] == type_str:
            result.append(sample)
    return result



def write_W(questions, answers, type_str, path='/home/ubuntu/11777-Project/common/tmp/'):
    with open(os.path.join(path + '{}{}.txt'.format(type_str, len(questions))), 'w') as out:
        for q, ans in zip(questions, answers):
            out.write(q)
            out.write('\n')
            for a in ans: # ans[0] is the true answer
                out.write(str(a))
                out.write('\n')
            out.write('\n')

            



def main():
    bdp_6w = BasicDataProvider(dataset_root='/home/ubuntu/11777-Project/data', 
                           dataset_name = 'dataset_v7w_telling.json')
    bdp_which = BasicDataProvider(dataset_root='/home/ubuntu/11777-Project/data', 
                            dataset_name = 'dataset_v7w_pointing.json')
    sevenWs = ['what', 'why', 'how', 'who', 'when', 'where', 'which']
    # Get 50 samples for each of 7W's questions
    sample_7Ws = []
    for type_str in sevenWs:
        bdp = bdp_which if type_str == 'which' else bdp_6w
        sample_7Ws.append(sample_by_type(bdp, type_str))
    # Get questions for each type
    sample_7Ws_questions = []
    sample_7Ws_answers = []
    for sample_W in sample_7Ws:
        questions = [x['question'] for x in sample_W]
        # Grab answers as [true_ans, false_ans1, false_ans2, false_ans3]
        answers = [[x['answer']] + x['multiple_choices'] for x in sample_W]
        sample_7Ws_questions.append(questions)
        sample_7Ws_answers.append(answers)
    # Output each W's questions to a text file
    for questions, answers, w_type in zip(sample_7Ws_questions, sample_7Ws_answers, sevenWs):
        write_W(questions, answers, w_type)



if __name__ == "__main__":
    main()