import os
import argparse 
import json

def count_hat_acc(dir, split_dir, topk=1) :
    accs = []
    for split in split_dir :
        with open(os.path.join(dir, split, "log.txt"), 'r') as f :
            data = f.read()
            data = json.loads(data.replace('\n', ''))
            if topk == 1 :
                accs.append(data["Final top-1"])
            elif topk == 5 :
                accs.append(data["Final Top-5"])
    
    acc = 0
    for a in accs :
        acc += float(a)
    
    print(acc / len(accs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--split_dir", nargs='+', type=str)
    args = parser.parse_args()
    
    accs = []
    for split_dir in args.split_dir :
        with open(os.path.join(args.dir, split_dir, "log.txt"), 'r') as f :
            data = f.read()
            data = json.loads(data.replace('\n', ''))
            accs.append(data["Final Top-5"])
    
    acc = 0
    for a in accs :
        acc += float(a)
    
    print(acc / len(accs))