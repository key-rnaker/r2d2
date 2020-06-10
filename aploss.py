import torch
import torch.nn as nn

class QAPLoss(nn.Module) :
    def __init__(self, nBin=25) :
        super().__init__()
        self.nBin = nBin    # number of Bin
        self.delta = 2/(nBin-1) # size of each Bin
        self.centerBin = torch.tensor([1-m*self.delta for m in range(nBin)])

    def cosdis(self, vi, vj) :
        """ 
        Args :
            - vi : Descriptors torch tensor of shape (batch_size, N, M)
            - vj : Descriptors torch tensor of shape (batch_size, N, M)
            N is number of Descriptor
            M is Descriptor dimension

        Return :
            - return : Cosine Similarity tensor of shape (batch_size, N)
        """
        cos = nn.CosineSimilarity(dim=2)
        return cos(vi, vj)

    def precision_recall(self, m, sim_q, labels) :
        # 0 <= m < nBin
        """
            - m : m-th bin
            - sim_q : Cosine Similarity between Descriptors tensor of shape (db_size)
            - labels : ground truth relavant between db and query tensor of shape (db_size)
        """
        db_size = sim_q.shape
        pDenominator = 0
        pNumerator = 1e-16

        # sim_q = [db_size, nBin]
        sim_q = sim_q.unsqueeze(-1).expand( -1, self.centerBin.shape[0])
        # centerBin = [db_size, nBin]
        centerBin = self.centerBin.unsqueeze(0).expand(sim_q.shape[0], -1)

        # soft_assignment = [db_size, nBin]
        soft_assignment = torch.clamp(1 - torch.abs(sim_q - centerBin )/self.delta, min=0)

        for n in range(m+1) :
            pDenominator += torch.dot(soft_assignment[:,n], labels[:].double())
            pNumerator += torch.dot(soft_assignment[:, n], torch.ones(db_size).double())
        
        recall = torch.dot(soft_assignment[:,m], labels[:].double())
        recall = torch.true_divide(recall, torch.sum(labels[:].double()))

        precision = torch.true_divide(pDenominator, pNumerator)

        # return shape [batch_size]
        return precision, recall

    def forward_one(self, X, Xs, label) :
        """
        Args :
            - X : query Descriptor torch tensor of shape (M)
            - Xs : db Descriptor torch tensor of shape (db_size, M)
            - label : ground truth relavant between db Descriptor and query Descriptor (db_size)
        """

        sim_q = self.cosdis(X.unsqueeze(0).unsqueeze(0).expand(-1,Xs.shape[0],-1), Xs.unsqueeze(0))

        APQ = []
        for m in range(self.nBin) :
            precision, recall = self.precision_recall(m, sim_q, label)
            APQ.append(precision*recall)

        return torch.mean(torch.stack(APQ))



    def forward(self, qX, dXs, labels ) :
        """
        Args :
            - qX : query Descriptor torch tensor of shape (batch_size, M)
            - dXs : db Descriptors torch tensor of shape (batch_size, db_size, M)
            - labels : ground truth relavant between dbDescriptors and query Descriptor (batch_size, db_size)
            M is Descriptor dimension
        """
        nBatch, Dim = qX.shape
        nDB = dXs.shape[1]

        # 1. get cosine similarity between all database Descriptors and query Descriptor
        # sim_q shape (batch_size, db_size)

        # 2. calculate AP per batch
        mAPQ = []
        for batch in range(nBatch) :
            APQ = self.forward_one(qX[batch], dXs[batch], labels[batch])
            mAPQ.append(APQ)

        return torch.mean(torch.stack(mAPQ))



