"""Learn an explicitly non-concept anchor-bias correction; freeze evidence encoder."""
import torch
from torch import nn
import torch.nn.functional as F
from erbc_io import ARMS, need
def freeze_evidence(head,s,f):
    captured=[]
    hook=head.body[1].register_forward_hook(lambda module,inputs,output:captured.append(output.detach().clone()))
    try:
        with torch.no_grad():z=head(s,f)
    finally:hook.remove()
    need(len(captured)==1 and captured[0].shape==(len(z['mass']),32),'LATENT_CAPTURE')
    return dict(latent=captured[0],responsibility=z['responsibility'].detach(),
      mass=z['mass'].detach(),prediction=z['prediction'].detach(),gate=s['gate'].detach())
class BiasReadout(nn.Module):
    def __init__(self,arm='joint'):
        super().__init__();need(arm in ARMS,'ARM');self.arm=arm
        self.norm=nn.LayerNorm(32)
        self.body=nn.Sequential(nn.Linear(104,32),nn.GELU(),nn.Linear(32,4))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)
    def forward(self,s):
        # Whitelisted state only: no label, ID, group, subject or gold KSA.
        h=self.norm(s['latent'].float());q=s['responsibility'].double();m=s['mass'].double()
        ell=q.clamp_min(1e-6).log();q0=ell.softmax(-1);v=q0.new_tensor([0.,.5,1.])
        c0=(m*(q0@v)).sum();p=s['prediction'].double();bias=p-c0
        pooled=(m.float()[:,None]*h).sum(0);avg=h.mean(0)
        fixed=torch.cat((q0.float(),m.float()[:,None],torch.stack((p,c0,bias,s['gate'].double())).float().expand(len(h),-1)),-1)
        x=torch.cat((h,pooled.expand(len(h),-1),avg.expand(len(h),-1),fixed),-1)
        raw=self.body(x).double()
        shift=6*raw[:,:3].tanh() if self.arm!='bias_only' else raw[:,:3]*0
        qnew=(ell+shift).softmax(-1)
        dq=(m*((qnew-q0)@v)).sum()
        # A separate scalar channel, NOT a KSA responsibility or calibrated confidence.
        db=(m*raw[:,3]).sum().tanh() if self.arm!='q_only' else raw[:,3].sum()*0
        unbounded=p+dq+db
        return dict(prediction=unbounded.clamp(0,1),unbounded=unbounded,reference_q=q0,
          responsibility=qnew,mass=m,delta_q=dq,delta_bias=db,retained_bias=bias,
          corrected_bias=bias+db,aggregate=c0+dq,shift=shift,
          q_KL=(m*(qnew*(qnew.clamp_min(1e-12).log()-q0.log())).sum(-1)).sum())
def ordinal(q,sup):
    if sup is None:return q.sum()*0
    mask=torch.tensor(sup['mask'],dtype=torch.bool);r=q.new_tensor(sup['responsibility'])
    need(mask.shape==(len(q),) and r.shape==q.shape,'LOCAL_SHAPE')
    if not bool(mask.any()):return q.sum()*0
    return .5*(q[mask].cumsum(-1)[:,:2]-r[mask].cumsum(-1)[:,:2]).abs().sum(-1).mean()
def loss_parts(head,clean,pred,y,sup):
    c=head(clean)
    terms=dict(clean_drift=.1*(c['prediction']-clean['prediction']).abs(),
      clean_local=.05*ordinal(c['responsibility'],sup['clean']),clean_KL=.01*c['q_KL'],
      clean_bias=.02*c['delta_bias'].abs())
    if pred is not None:
        z=head(pred);e=(z['prediction']-y).abs();old=(pred['prediction']-y).abs()
        terms.update(MAE=e,excess=.1*F.relu(e-old),pred_drift=.02*(z['prediction']-pred['prediction']).abs(),
          pred_local=.05*ordinal(z['responsibility'],sup['predicted']),pred_KL=.01*z['q_KL'],
          pred_bias=.02*z['delta_bias'].abs())
    return sum(terms.values()),terms
