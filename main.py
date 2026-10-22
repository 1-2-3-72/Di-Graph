from __future__ import print_function

import argparse
import os
import random
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.core_utils import train_student, train_teacher


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_topk(value):
    if value is None or value == "":
        return None
    return tuple(float(x) for x in value.split(",") if x.strip())


def write_summary(path, folds, vals):
    vals = [float(v) for v in vals]
    df = pd.DataFrame({'folds': list(folds), 'val_cindex': vals})
    if vals:
        df.loc[len(df)] = ['mean', float(np.mean(vals))]
        df.loc[len(df)] = ['std', float(np.std(vals))]
    df.to_csv(path, index=False)


def build_dataset(args):
    combined_study = '_'.join(args.split_dir.split('_')[:2])
    csv_path = f'./dataset_csv_sig/{combined_study}_all_clean.csv.zip'
    return Generic_MIL_Survival_Dataset(
        csv_path=csv_path,
        mode=args.mode,
        apply_sig=args.apply_sig,
        data_dir=args.data_root_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col='survival_months',
        ignore=[])


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)

    dataset = build_dataset(args)
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
    args.task_type = 'survival'
    args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
    assert os.path.isdir(args.split_dir), args.split_dir

    print('################# Settings ###################')
    for key, val in sorted(vars(args).items()):
        print(f'{key}: {val}')

    teacher_vals = []
    student_vals = []

    if args.stage in ('teacher', 'both'):
        for i in folds:
            start_t = timer()
            seed_torch(args.seed)
            train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path=f'{args.split_dir}/splits_{i}.csv')
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
            args.omic_sizes = train_dataset.omic_sizes
            print('omic_sizes shape', args.omic_sizes)
            val = train_teacher((train_dataset, val_dataset), int(i), args)
            teacher_vals.append(val)
            print('Teacher Fold %d Time: %f seconds' % (i, timer() - start_t))
        write_summary(os.path.join(args.results_dir, 'summary_teacher.csv'), folds, teacher_vals)
        print('The final teacher performance mean {} and std {}'.format(np.mean(teacher_vals), np.std(teacher_vals)))

    if args.stage in ('student', 'both'):
        for i in folds:
            start_t = timer()
            seed_torch(args.seed)
            train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path=f'{args.split_dir}/splits_{i}.csv')
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
            args.omic_sizes = train_dataset.omic_sizes
            print('omic_sizes shape', args.omic_sizes)
            val = train_student((train_dataset, val_dataset), int(i), args)
            student_vals.append(val)
            print('Student Fold %d Time: %f seconds' % (i, timer() - start_t))
        write_summary(os.path.join(args.results_dir, 'summary_student.csv'), folds, student_vals)
        print('The final student performance mean {} and std {}'.format(np.mean(student_vals), np.std(student_vals)))


parser = argparse.ArgumentParser(description='G-HANet progressive teacher/student survival experiments')
parser.add_argument('--data_root_dir', type=str, required=True, help='Dataset feature directory containing pt_files')
parser.add_argument('--split_dir', type=str, default='tcga_brca')
parser.add_argument('--stage', type=str, choices=['teacher','student','both'], default='both')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--teacher_ckpt_dir', type=str, default='', help='Directory containing teacher_final_fold checkpoints for student training')
parser.add_argument('--which_splits', type=str, default='5foldcv')
parser.add_argument('--model_type', type=str, default='mcat')
parser.add_argument('--mode', type=str, default='coattn')
parser.add_argument('--fusion', type=str, default='concat')
parser.add_argument('--apply_sig', action='store_true', default=True)
parser.add_argument('--apply_sigfeats', action='store_true', default=True)
parser.add_argument('--drop_out', action='store_true', default=True)
parser.add_argument('--model_size_wsi', type=str, default='small')
parser.add_argument('--model_size_omic', type=str, default='small')
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--alpha_layers', type=int, default=3)
parser.add_argument('--teacher_topk', type=str, default='', help='Comma-separated teacher top-k retention ratios')
parser.add_argument('--student_topk', type=str, default='', help='Comma-separated student top-k retention ratios')
parser.add_argument('--opt', type=str, choices=['adam','sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gc', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--bag_loss', type=str, choices=['svm','ce','ce_surv','nll_surv','cox_surv'], default='nll_surv')
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--alpha_surv', type=float, default=0.0)
parser.add_argument('--reg_type', type=str, default='None')
parser.add_argument('--lambda_reg', type=float, default=0)
parser.add_argument('--lambda1', type=float, default=0.3)
parser.add_argument('--lambda2', type=float, default=0.3)
parser.add_argument('--lambda3', type=float, default=0.3)
parser.add_argument('--lambda4', type=float, default=0.3)
parser.add_argument('--weighted_sample', action='store_true', default=True)
parser.add_argument('--early_stopping', action='store_true', default=False)

args = parser.parse_args()
args.teacher_topk = parse_topk(args.teacher_topk)
args.student_topk = parse_topk(args.student_topk)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_torch(args.seed)

if __name__ == '__main__':
    start = timer()
    print('\nLearning Rate set to: {}\n'.format(args.lr))
    main(args)
    print('finished!')
    print('\nLearning Rate set to: {}\n'.format(args.lr))
    print('Script Time: %f seconds' % (timer() - start))
