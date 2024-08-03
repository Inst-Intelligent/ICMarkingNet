'''
Code for paper ICMarkingNet: An Ultra-Fast and Streamlined 
Deep Model for IC Marking Inspection
[Latest Update] 3 Aug 2024
'''

import torch
import time
import editdistance 

from utils.ctc_decoder import ctc_decode
import roi

def label2Char(label):
    return "".join([roi.LABEL2CHAR[x] for x in label])

def evaluate(model, data_loader, criterion):
    model.eval()

    tot_image_count = 0
    tot_word_count = 0
    tot_loss = 0
    tot_correct = 0
    tot_ed = 0
    tot_reallen = 0
    tot_predlen = 0
    wrong_cases = []

    start_time = time.time()

    for index, inputs in enumerate(data_loader):
        with torch.no_grad():
            results, labels, words_roi = model(inputs)
            loss, details = criterion(results, labels)
            
            _, _, marking_results = results
            _, _, (targets, target_lengths) = labels
            preds = ctc_decode(marking_results.detach(), method='greedy', beam_size=10)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_image_count += len(inputs)
            tot_word_count += len(target_lengths)
            tot_loss += loss.item()
            target_length_counter = 0

            n = 0
            sample_cases = []
            for pred, target_length in zip(preds, target_lengths):
                real = reals[
                    target_length_counter : target_length_counter + target_length
                ]
                real = label2Char(real)
                pred = label2Char(pred)
                target_length_counter += target_length
                sample_cases.append((real, pred))
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

                ed = editdistance.eval(real, pred) - max(0, len(pred) - len(real))
                tot_ed += ed
                tot_reallen += len(real)
                tot_predlen += len(pred)

                n+=1

    evaluation = {
        "acc": tot_correct / tot_word_count,
        "wrong_cases": wrong_cases,
        "rec": 1 - (tot_ed / tot_reallen),
        "prec": 1 - (tot_ed / tot_predlen),
        "sed": tot_ed / tot_image_count,
        "time": time.time() - start_time
    }
    return evaluation

def print_results(results):
    print("*****************")
    print("** Wrong cases **")
    for gt, pred in results['wrong_cases']:
        print(f'{gt.upper()} -> {pred.upper()}')
    print("*****************")
    print(
        "Accurate: {acc:.4f}\n"
        "Recall: {rec:.4f}\n"
        "Precision: {prec:.4f}\n"
        "SED: {sed:.4f}\n"
        "Elasped time: {time: .4f}s"
        .format(**results)
    )
    print("*****************")
