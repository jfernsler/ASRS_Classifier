import argparse, sys, random

from jf_nlp.nlp_dataloaders import ASRSTestLoader
from asrs_multi_infer import infer_one, infer_many

def print_dict(data_dict):
    for k,v in data_dict.items():
        print(f'{k}:', end='', flush=True)
        for n, char in enumerate(str(v)):
            if n % 60 == 0:
                print('\n\t', end='', flush=True)
            print(char, end='', flush=True)
        print()
        print()

def main(args):
    if args.test_dataset:
        print('*'*10, ' Test Dataset ', '*'*10, '\n')

        test = ASRSTestLoader()
        rand_idx = random.randint(0, len(test.data))
        data_dict = test.asrs.iloc[rand_idx].to_dict()
        print()
        print(f'Index: {rand_idx}')
        print_dict(data_dict)

    elif args.predict_one:
        print('*'*10, ' Predict One ', '*'*10)
        infer_one()
        
    elif args.predict_many:
        print('*'*10, ' Predict Many ', '*'*10)
        infer_many(args.predict_many)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CS614 Assignment 3 - Language Classifier')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--test_dataset', 
                        action='store_true', help='View a randomly selected data member from the Test Dataset')
    group.add_argument('-po', '--predict_one',
                        action='store_true', help='Predict one randomly selected narrative')
    group.add_argument('-pm', '--predict_many', const=10,
                        action='store_const', help='Predict many randomly selected narratives')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    main(args)