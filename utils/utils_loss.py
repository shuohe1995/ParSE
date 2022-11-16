import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss1(nn.Module):
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
    def forward(self,outputs,index):
        all_confidence=torch.cat([self.confidence[index],self.confidence[index]])
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * all_confidence
        loss = - ((final_outputs).sum(dim=1)).mean()
        return loss
    def confidence_update(self,outputs1,outputs2,batchY,batch_index):
        with torch.no_grad():
            sm_outputs = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
            sm_outputs *= batchY
            new_batch_confidence = sm_outputs / sm_outputs.sum(dim=1, keepdim=True)
            self.confidence[batch_index]=new_batch_confidence
        return None


class partial_loss2(nn.Module):
    def __init__(self, confidence):
        super().__init__()
        self.confidence = confidence
    def forward(self,outputs,index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index]
        loss = - ((final_outputs).sum(dim=1)).mean()
        return loss

        return loss
    def confidence_update(self,batch_outputs,batchY,batch_index):
        with torch.no_grad():
            sm_outputs=torch.softmax(batch_outputs, dim=1)
            new_batch_confidence =sm_outputs * batchY
            new_batch_confidence = new_batch_confidence / new_batch_confidence.sum(dim=1, keepdim=True)
            self.confidence[batch_index] = new_batch_confidence
        return

class semantic_loss(nn.Module):
    def __init__(self,sigma):
        super().__init__()
        self.sigma = sigma
    def forward(self,feature,class_embedding,batch_confidence):
        all_similarity = torch.mm(feature, class_embedding.t())

        weighted_positive_similarity = (all_similarity * batch_confidence).sum(dim=1)

        negative_mask=(batch_confidence==0).float()

        negative_similarity = all_similarity * negative_mask

        loss=torch.clamp((self.sigma-weighted_positive_similarity.unsqueeze(dim=1) + negative_similarity) * negative_mask,min=0).sum(dim=1).mean()

        return loss