"""Paired continuation: change reference targets only, not architecture."""
import math
import torch
import torch.nn.functional as F
from erbc_head import ordinal, loss_parts as original_loss

ARMS=('old_reference','success_reference')
KEYS=('overall','computer_science')
def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def detach_output(z):
    return {k:v.detach().clone() for k,v in z.items() if torch.is_tensor(v)}
def new_loss(head,clean,pred,y,sup,teacher_clean,teacher_pred,arm):
    need(arm in ARMS,'ARM')
    if arm=='old_reference':return original_loss(head,clean,pred,y,sup)
    c=head(clean);tc=teacher_clean
    def kl(z,t):
        q=z['responsibility'];ref=t['responsibility'].clamp_min(1e-12)
        return (z['mass']*(q*(q.clamp_min(1e-12).log()-ref.log())).sum(-1)).sum()
    terms=dict(clean_drift=.1*(c['prediction']-tc['prediction']).abs(),
      clean_local=.05*ordinal(c['responsibility'],sup['clean']),
      clean_KL=.01*kl(c,tc),clean_bias=.02*(c['delta_bias']-tc['delta_bias']).abs())
    if pred is not None:
        need(teacher_pred is not None,'TEACHER_REQUIRED')
        z=head(pred);t=teacher_pred;e=(z['prediction']-y).abs();old=(t['prediction']-y).abs()
        terms.update(MAE=e,excess=.1*F.relu(e-old),
          pred_drift=.02*(z['prediction']-t['prediction']).abs(),
          pred_local=.05*ordinal(z['responsibility'],sup['predicted']),
          pred_KL=.01*kl(z,t),pred_bias=.02*(z['delta_bias']-t['delta_bias']).abs())
    return sum(terms.values()),terms