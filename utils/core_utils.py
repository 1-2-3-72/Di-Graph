from argparse import Namespace
import os
import torch
from models.model_coattn import Teacher_G_HANet_Surv, Student_G_HANet_Surv
from utils.utils import *
from utils.coattn_train_utils import train_loop_teacher, train_loop_student, validate_survival

def train_teacher(datasets: tuple, cur: int, args: Namespace):
    print('\nTraining Fold {}!'.format(cur))
    writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')


    print('\nInit Teacher Model...', end=' ')
    model_dict = {'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes, 'fusion': args.fusion,
                  'model_size_wsi': args.model_size_wsi, 'model_size_omic': args.model_size_omic, 'path_input_dim': args.path_input_dim, 'alpha_layers': args.alpha_layers, 'teacher_topk': args.teacher_topk}
    teacher_model = Teacher_G_HANet_Surv(**model_dict).to(device)
    print('Done!')


    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(teacher_model, args)
    print('Done!')



    teacher_c_index_best = 0
    teacher_best_epoch = -1
    for epoch in range(args.max_epochs):
        train_loop_teacher(
            epoch, teacher_model, train_loader, optimizer, loss_fn, 
            args.gc)
        
        val_loss ,val_index = validate_survival(epoch, teacher_model, val_loader, loss_fn)

        if val_index > teacher_c_index_best:
            teacher_c_index_best = val_index
            teacher_best_epoch = epoch
            torch.save({
                'student_model': teacher_model.state_dict(),
            }, os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))
        print(f"\033[91mEpoch {epoch}: Best C-index so far is {teacher_c_index_best} at epoch {teacher_best_epoch}\033[0m")

    torch.save(teacher_model.state_dict(), os.path.join(args.results_dir,  f"teacher_final_fold{cur}.pt"))
    return teacher_c_index_best

def train_student(datasets: tuple, cur: int, args: Namespace):
   
    teacher_models = []
    for i in range(args.k_start if args.k_start != -1 else 0, args.k_end if args.k_end != -1 else args.k):  
        model_dict = {'omic_sizes': args.omic_sizes, 'fusion': args.fusion, 'n_classes': args.n_classes,
                  'model_size_wsi': args.model_size_wsi, 'model_size_omic': args.model_size_omic, 'path_input_dim': args.path_input_dim, 'alpha_layers': args.alpha_layers, 'teacher_topk': args.teacher_topk}
        teacher_model = Teacher_G_HANet_Surv(**model_dict).to(device)
        teacher_ckpt_dir = args.teacher_ckpt_dir if getattr(args, 'teacher_ckpt_dir', '') else args.results_dir
        teacher_model.load_state_dict(torch.load(os.path.join(teacher_ckpt_dir, f"teacher_final_fold{i}.pt")))
        teacher_model.eval()  
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_models.append(teacher_model)
       
    print("\nAll Pre-Trained Teacher Models Loaded! Now Training Student Models...")

    print('\nTraining Fold {}!'.format(cur))
    
    writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')
    
    print('\nInit Student Model...', end=' ')
    student_model = Student_G_HANet_Surv(model_size_wsi=args.model_size_wsi, n_classes=args.n_classes, path_input_dim=args.path_input_dim, alpha_layers=args.alpha_layers, student_topk=args.student_topk).to(device)
    print('Done!') 

    
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(student_model, args)
    print('Done!')


    student_c_index_best = 0
    student_best_epoch = -1
    for epoch in range(args.max_epochs):
        train_loop_student(epoch, teacher_models, student_model, train_loader, optimizer, loss_fn, args.gc, args.lambda1, args.lambda2, args.lambda3, args.lambda4)
        val_loss, val_index = validate_survival(epoch, student_model, val_loader, loss_fn)
        if val_index > student_c_index_best:
            student_c_index_best = val_index
            student_best_epoch = epoch
            torch.save({
                'student_model': student_model.state_dict(),
            }, os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))
        print(f"\033[91mEpoch {epoch}: Best C-index so far is {student_c_index_best} at epoch {student_best_epoch}\033[0m")

        
    torch.save(student_model.state_dict(), os.path.join(args.results_dir,  f"student_checkpoint_fold{cur}.pt"))

    return student_c_index_best

 
