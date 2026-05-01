import torch
import torch.nn as nn

class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0, normalize_feature = False):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square
        self.normalize_feature = normalize_feature
        self.dist = nn.CosineSimilarity(dim=0)

    def forward(self, anchor, positive, weight_mask, *neglist):
        batch = anchor.size(0)
        label_V = torch.ones(batch,1).to('cuda')
        loss_2an = torch.zeros(len(neglist)).to('cuda')
        loss_2pn = torch.zeros(len(neglist)).to('cuda')
        # loss_ap = torch.pairwise_distance(anchor, positive)
        loss_ap = self.dist(anchor, positive)
        if self.normalize_feature:
            anchor = normalize(anchor, axis=-1)
            positive = normalize(positive, axis=-1)

        for i in range(len(neglist)):
            if self.normalize_feature:
                neglist[i] = normalize(neglist[i], axis=-1)
            is_pos = pair_similarity(label_V, weight_mask[:,i].unsqueeze(1))
            is_neg = pair_similarity(label_V, 1-weight_mask[:,i].unsqueeze(1))
            an_dist_mat = pdist_torch(anchor, neglist[i])
            pn_dist_mat = pdist_torch(positive, neglist[i])

            weights_anp = softmax_weights(an_dist_mat, is_pos)
            weights_ann = softmax_weights(-an_dist_mat, is_neg)
            Afurthest_positive = torch.sum(an_dist_mat * weights_anp, dim=1)
            Aclosest_negative  = torch.sum(an_dist_mat * weights_ann, dim=1)

            weights_pnp = softmax_weights(pn_dist_mat, is_pos)
            weights_pnn = softmax_weights(-pn_dist_mat, is_neg)
            Pfurthest_positive = torch.sum(pn_dist_mat * weights_pnp, dim=1)
            Pclosest_negative  = torch.sum(pn_dist_mat * weights_pnn, dim=1)

            if self.square ==0:
                y = Pfurthest_positive.new().resize_as_(Pfurthest_positive).fill_(1)
                loss_2an[i] = self.ranking_loss(self.gamma*(Aclosest_negative - Afurthest_positive), y)
                loss_2pn[i] = self.ranking_loss(self.gamma*(Pclosest_negative - Pfurthest_positive), y)
            else:
                diff_powp = torch.pow(Pfurthest_positive - Pclosest_negative, 2) * self.gamma
                diff_powp =torch.clamp_max(diff_powp, max=88)
                # Compute ranking hinge loss
                y1p = (Pfurthest_positive > Pclosest_negative).float()
                y2p = y1p - 1
                yp = -(y1p + y2p)
                
                loss_2pn[i] = self.ranking_loss(diff_powp, yp)

                diff_powa = torch.pow(Afurthest_positive - Aclosest_negative, 2) * self.gamma
                diff_powa =torch.clamp_max(diff_powa, max=88)

                # Compute ranking hinge loss
                y1 = (Afurthest_positive > Aclosest_negative).float()
                y2 = y1 - 1
                y = -(y1 + y2)
                
                loss_2an[i] = self.ranking_loss(diff_powa, y)
        loss_apn = loss_ap.sum()/ batch + loss_2pn + loss_2an
        
        loss = torch.clamp(loss_apn, min=0).sum()/2

        return loss

def pair_similarity(x, y):
    '''
    x: n * dx
    y: m * dy
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    ps = torch.eq(x, y).squeeze(2)
    ps = ps.float()
    # ps -= (ps == 0.).float()
    return ps

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx  

class lift_4struct(nn.Module):
    def __init__(self, alpha, multi, margin=0.0):
        super(lift_4struct, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.multi = multi
        self.dist = nn.CosineSimilarity(dim=0)

    def forward(self, anchor, positive, weight_mask, *neglist):
        batch = anchor.size(0)
        loss_2an = torch.zeros(batch,self.multi).cuda()
        loss_2pn = torch.zeros(batch,self.multi).cuda()
        coff_2an = torch.zeros(batch,self.multi).cuda()
        coff_2pn = torch.zeros(batch,self.multi).cuda()
        #loss_ap = torch.pairwise_distance(anchor, positive)
        loss_ap = torch.pairwise_distance(anchor, positive)
        Mloss_ap = torch.clamp(loss_ap, min=self.margin)

        for i in range(len(neglist)):            
            loss_2an[:,i]= torch.pairwise_distance(anchor, neglist[i])
            loss_2pn[:,i]= torch.pairwise_distance(positive, neglist[i])
            coff_2an[:,i]= torch.exp(Mloss_ap-torch.pairwise_distance(anchor, neglist[i]))
            coff_2pn[:,i]= torch.exp(Mloss_ap-torch.pairwise_distance(positive, neglist[i]))

        loss_an = torch.mul(torch.exp(self.alpha - loss_2an), coff_2an)
        dist_an= torch.mul(loss_an, weight_mask)
   
        loss_pn = torch.mul(torch.exp(self.alpha - loss_2pn), coff_2pn)
        dist_pn = torch.mul(loss_pn, weight_mask)
        loss_apn = Mloss_ap + 10*torch.log(1 + 10*(torch.max(dist_an,dim=1)[0] + torch.max(dist_pn,dim=1)[0])).to('cuda')# + torch.log(loss_exp_anpn)
        loss_apn = torch.clamp(loss_apn, min=0).sum() / (2*batch)

        return loss_apn


class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='cos'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2):
 
        for i in range(1):
          center1 = torch.mean(feat1, dim=0)
          center2 = torch.mean(feat2, dim=0)

          # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
        if self.dist_type == 'l2' or self.dist_type == 'l1':
            if i == 0:
              # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
                # dist = max(abs(self.margin - self.dist(center1, center2)), abs(self.margin - self.dist(center2, center3)), abs(self.margin - self.dist(center1, center3)))
                dist = max(0, abs(self.margin - self.dist(center1, center2)))

        elif self.dist_type == 'cos':
            dist = max(0, 1-self.dist(center1, center2))
        return dist
 
 
class MMDLoss(nn.Module):

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='cos'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2):
        feat1 = normalize(feat1)
        feat2 = normalize(feat2)
 
        center1 = torch.mean(feat1, dim=0)
        center2 = torch.mean(feat2, dim=0)

          # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
        if self.dist_type == 'l2' or self.dist_type == 'l1':
              # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
                # dist = max(abs(self.margin - self.dist(center1, center2)), abs(self.margin - self.dist(center2, center3)), abs(self.margin - self.dist(center1, center3)))
            dist = max(0, abs(self.margin - self.dist(center1, center2)))

        elif self.dist_type == 'cos':
            dist = max(0, 1-self.dist(center1, center2))
        return dist

    def normalize(x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x 
        
class lift_struct(nn.Module):
    def __init__(self, alpha, multi):
        super(lift_struct, self).__init__()
        self.alpha = alpha
        self.multi = multi

    def forward(self, anchor, positive, neglist):
        batch = anchor.size(0)
        D_ij = torch.pairwise_distance(anchor, positive)
        D_in = torch.zeros(batch,self.multi)
        D_jn = torch.zeros(batch,self.multi)
        for i in range(self.multi):
            a = torch.pairwise_distance(anchor, neglist[i])
            D_in[:,i]= torch.exp(self.alpha - a)
            b = torch.pairwise_distance(positive, neglist[i])
            D_jn[:,i]= torch.exp(self.alpha - b)
        D_n = D_in.max(1)[0] + D_jn.max(1)[0]
        J = torch.log(D_n).to('cuda') + D_ij
        J = torch.clamp(J, min=0)
        loss = J.sum() / (2 * batch)
        return loss

class re_triplet(nn.Module):
    def __init__(self, margin):
        super(re_triplet, self).__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(self.margin)

    def forward(self, anchor, positive, n_list):
        loss = 0.0
        for i in range(len(n_list)):
            loss += self.loss(anchor, positive, n_list[i])
        loss = loss / len(n_list)
        return loss

        
"""
if __name__ == '__main__':
    L = lift_struct(1.0, 1)
    # L = RankList(1.2,0.4,1,2)
    # L = n_pair(2)
    anchor = torch.randn(64, 128)
    positive = torch.randn(64, 128)
    negative1 = torch.randn(64, 128)
    # negative2 = torch.randn(64,128)
    neglist = []
    # neglist.append(negative1)
    # neglist.append(negative2)
    loss = L(anchor, positive, negative1)
    print(loss)
    # print(loss)
    # count = distance_acc(anchor,positive,neglist)
    # print(count)
"""