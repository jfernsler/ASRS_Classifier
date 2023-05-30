import argparse, sys, random

from jf_nlp.nlp_dataloaders import ASRSTestLoader
from asrs_infer import infer_one, infer_many

def main(args):
    if args.test_dataset:
        print('*'*10, ' Test Dataset ', '*'*10, '\n')
        idx = int(args.test_dataset)
        test = ASRSTestLoader()
        if idx == -1:
            idx = random.randint(0, len(test.data))
        data_dict = test.asrs.iloc[idx].to_dict()
        print()
        print(f'Index: {idx}')
        print_dict(data_dict)

    elif args.predict_one:
        print('*'*10, ' Predict One ', '*'*10)
        pred_idx = int(args.predict_one)
        if args.predict_one == -1:
            infer_one()
        else:
            infer_one(pred_idx)
        
    elif args.predict_many:
        print('*'*10, ' Predicting {0} Narratives '.format(args.predict_many), '*'*10)
        pred_count = int(args.predict_many)
        infer_many(pred_count)


def print_dict(data_dict):
    for k,v in data_dict.items():
        print(f'{k}:', end='', flush=True)
        for n, char in enumerate(str(v)):
            if n % 60 == 0:
                print('\n\t', end='', flush=True)
            print(char, end='', flush=True)
        print()
        print()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CS614 Assignment 3 - ASRS Language Classifier')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--test_dataset', nargs='?', const='-1',
                        action='store', help='Given no index number, view a randomly selected data member from the Test Dataset')
    group.add_argument('-po', '--predict_one', nargs='?', const='-1',
                        action='store', help='Given no index number, predict a random narrative. Given an index number, predict that narrative.')
    group.add_argument('-pm', '--predict_many', const='10', nargs='?',
                        action='store', help='Predict many randomly selected narratives')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)