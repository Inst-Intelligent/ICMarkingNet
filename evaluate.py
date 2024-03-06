import torch
import time
import editdistance 

from utils.ctc_decoder import ctc_decode
import model.roi as roi

def label2Char(label):
    return "".join([roi.LABEL2CHAR[x] for x in label])

def evaluate(
    model, 
    val_loader,  
    criterion,
    batch_size,
    max_iter=None,
    decode_method="beam_search",
    beam_size=10,
    types = None
):
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

    for index, inputs in enumerate(val_loader):
        with torch.no_grad():

            if types is None:
                results, labels, words_roi = model(inputs)
            
            else:
                results, labels, words_roi = model(inputs)
                names = [info.name for info in inputs[-1]]

                words_types = []
                words_from_sample = words_roi[:,0].int().cpu().numpy()
                for i in range(words_roi.size(0)):
                    name = names[words_from_sample[i].item()]
                    if name in types:
                        words_types.append(types[name])
                    else:
                        words_types.append(0)
               
        
            loss, details = criterion(results, labels)
            

            _, _, marking_results = results
            _, _, (targets, target_lengths) = labels
            preds = ctc_decode(marking_results.detach(), method='greedy', beam_size=10)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_image_count += batch_size
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
        "loss": tot_loss / tot_image_count,
        "acc": tot_correct / tot_word_count,
        "wrong_cases": wrong_cases,
        "rec": 1 - (tot_ed / tot_reallen),
        "prec": 1 - (tot_ed / tot_predlen),
        "sed": tot_ed / tot_image_count,
        "time": time.time() - start_time
    }
    return evaluation