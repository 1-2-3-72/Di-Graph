from utils.utils import *
import numpy as np
from sksurv.metrics import concordance_index_censored
from models.model_coattn import Teacher_G_HANet_Surv
import torch

def distillation_loss(student_logits, teacher_logits, temperature):
    # 教师和学生的logits均除以温度参数，并计算概率分布
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # 计算KL散度损失，并乘以温度平方以平衡梯度
    loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    return loss

def train_loop_teacher(epoch, teacher_model, train_loader, optimizer, loss_fn=None, gc=16, omic_mask=None):

    teacher_model.train() 
    train_loss_surv = 0.0
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(train_loader):

        data_WSI= data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        event_time = event_time.to(device)
        label = label.to(device)
        c = c.to(device)

        hazards, S, _, _, _, _, _, _, _, _, _= teacher_model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, 
            x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)     

        loss_survival = loss_fn(hazards, S, label, c)
        train_loss_surv += loss_survival.item()

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)
        all_censorships.append(c.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(batch_idx, label.detach().cpu().item(), float
                                                                                     (event_time.detach().cpu().item()), float(risk)))

        (loss_survival / gc).backward()
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    if len(all_risk_scores) == 0:
        print(f"Warning: No valid risk scores computed in Epoch {epoch}")
        train_c_index = float('nan')
    else:
        train_c_index = concordance_index_censored(
            (1 - np.array(all_censorships)).astype(bool),
            np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08
        )[0]

    print \
        (f"Epoch {epoch}: Teacher_Train_Loss_Survival = {train_loss_surv / len(train_loader):.4f}, Train_C-Index = {train_c_index:.4f}")

def train_loop_student(epoch, teacher_models, student_model, train_loader, optimizer, loss_fn=None, gc=16, lambda1=0.3, lambda2=0.3, lambda3=0.3, lambda4=0.3):
 
    student_model.train()
    student_loss_surv, student_loss = 0., 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(train_loader):

        data_WSI, label, c= data_WSI.to(device), label.to(device), c.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        event_time = event_time.to(device)
      
        with torch.no_grad():

            teacher_logits_list = []
            pfa_values_list = []
            gene_tokens_with_cls_list= []
            fusion_list = []
            gene_repr_list = []
            patch_repr_list = []
            gp_weights_list = []
            pp_weights_list = []
            gg_weights_list = []

            for idx, teacher_model in enumerate(teacher_models):
                hazards, S, logits, pfa_values, gene_tokens_with_cls, fusion, gene_repr, patch_repr, gp_weights, pp_weights, gg_weights = teacher_model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                                x_omic3=data_omic3, x_omic4=data_omic4,
                                                                x_omic5=data_omic5,
                                                                x_omic6=data_omic6)
                
                teacher_logits_list.append(logits)
                pfa_values_list.append(pfa_values)
                gene_tokens_with_cls_list.append(gene_tokens_with_cls)
                fusion_list.append(fusion)
                gene_repr_list.append(gene_repr)
                patch_repr_list.append(patch_repr)
                gp_weights_list.append(gp_weights)
                pp_weights_list.append(pp_weights)
                gg_weights_list.append(gg_weights)

            teacher_logits_avg = torch.mean(torch.stack(teacher_logits_list), dim=0)
            gene_tokens_with_cls_avg = torch.mean(torch.stack(gene_tokens_with_cls_list), dim=0)
            pfa_values_avg = torch.mean(torch.stack(pfa_values_list), dim=0)
            fusion_avg = torch.mean(torch.stack(fusion_list), dim=0)
            gene_repr_avg = torch.mean(torch.stack(gene_repr_list), dim=0)
            patch_repr_avg = torch.mean(torch.stack(patch_repr_list), dim=0)
            gp_weights_avg = torch.mean(torch.stack(gp_weights_list), dim=0)
            pp_weights_avg = torch.mean(torch.stack(pp_weights_list), dim=0)
            gg_weights_avg = torch.mean(torch.stack(gg_weights_list), dim=0)

        hazards_s, S_s, logits_s, pfa_values_s, anchor_tokens,  fusion_s, gene_repr_s, patch_repr_s, ap_weights_s, pp_weights_s, aa_weights_s = student_model(x_path=data_WSI)
        #生存loss
        loss_survival = loss_fn(hazards_s, S_s, label, c)
        student_loss_surv += loss_survival.item()
     
        #gene token
        gene_loss = F.mse_loss(anchor_tokens, gene_tokens_with_cls_avg.detach())

        #assign loss
        ass_loss = F.mse_loss(pfa_values_s, pfa_values_avg.detach())

        # ==== 图节点loss ====
        loss_node_patch = F.mse_loss(patch_repr_s, patch_repr_avg.detach())
        loss_node_anchor = F.mse_loss(gene_repr_s, gene_repr_avg.detach())
        loss_node = loss_node_patch + loss_node_anchor

        # ==== 图 attention loss ====
        loss_edge_ap = F.mse_loss(ap_weights_s, gp_weights_avg.detach())
        loss_edge_pp = F.mse_loss(pp_weights_s, pp_weights_avg.detach())
        loss_edge_aa = F.mse_loss(aa_weights_s, gg_weights_avg.detach())
        loss_edge = loss_edge_ap + loss_edge_pp + loss_edge_aa

        #图结构整体loss
        loss_HGT = loss_node + loss_edge 

        #fusion loss
        cosine_similarity1 = F.cosine_similarity(fusion_s, fusion_avg.detach(), dim=1).mean()
        loss_fusion = 1 - cosine_similarity1 

        #logits loss
        kd_loss = distillation_loss(logits_s, teacher_logits_avg.detach(), temperature=3.0)
        loss_results = loss_fusion + kd_loss

        loss = loss_survival + lambda1 * gene_loss + lambda2 * ass_loss + lambda3 * loss_HGT + lambda4 * loss_results

        student_loss += loss.item()

        risk = -torch.sum(S_s, dim=1).detach().cpu().numpy()
        all_risk_scores.append(risk)
        all_censorships.append(c.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(
                'batch {}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(batch_idx, label.detach().cpu().item(),
                                                                                float(event_time.detach().cpu().item()),
                                                                                float(risk)))

        (loss / gc).backward()
        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    if len(all_risk_scores) == 0:
        print(f"Warning: No valid risk scores computed in Epoch {epoch}")
        train_c_index = float('nan')
    else:
        train_c_index = concordance_index_censored(
            (1 - np.array(all_censorships)).astype(bool),
            np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08
        )[0]

    print(f"Epoch {epoch}: Student_Train_Loss_Survival = {student_loss_surv / len(train_loader):.4f}, Student_Train_Loss = {student_loss / len(train_loader):.4f}, Student_Train C-Index = {train_c_index:.4f}")

def validate_survival(epoch, model, val_loader, loss_fn=None, omic_mask=None):
    model.eval()
    total_val_loss = 0.0

    all_risk_scores = np.zeros((len(val_loader)))
    all_censorships = np.zeros((len(val_loader)))
    all_event_times = np.zeros((len(val_loader)))

    with torch.no_grad():
        for batch_idx, (
        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
        c) in enumerate(val_loader):
            data_WSI = data_WSI.to(device)
            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)

            label = label.type(torch.LongTensor).to(device)
            c = c.to(device)
            event_time = event_time.to(device)
          
            if isinstance(model, Teacher_G_HANet_Surv):
                hazards, S, _, _, _, _, _, _, _, _, _= model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
                                         x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            else:
                hazards, S, _, _, _, _, _, _, _, _, _= model(x_path=data_WSI)
            val_loss = loss_fn(hazards, S, label, c)
            total_val_loss += val_loss.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

    avg_val_loss = total_val_loss / len(val_loader)

    if np.all(np.array(all_censorships) == 1):  
        print("Warning: All validation samples are censored! Skipping C-Index calculation.")
        val_index = float("nan")
    else:
        val_index = concordance_index_censored((1 - np.array(all_censorships)).astype(bool),
                                               np.array(all_event_times),
                                               np.array(all_risk_scores), tied_tol=1e-08)[0]

    print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}, C-Index: {val_index :.4f}")

    return avg_val_loss, val_index

