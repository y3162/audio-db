"""Checkpoint/resume probe for a deterministic CPU training stream."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from tests.training.four_process.compare import digest

def run(path, stop, resume):
    torch.set_num_threads(1); torch.manual_seed(42)
    model=torch.nn.Linear(8,8); optim=torch.optim.Adam(model.parameters(),lr=.001); sched=torch.optim.lr_scheduler.StepLR(optim,1,.9); position=0
    if resume:
        state=torch.load(path,weights_only=True); model.load_state_dict(state['model'],strict=True); optim.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); torch.set_rng_state(state['rng']); position=state['data_position']
    while position < stop:
        # Simulate a random transform and data order dependent on the saved data position.
        x=torch.randn(1,8)+position; target=torch.ones(1,8)*position
        optim.zero_grad(); (model(x)-target).square().mean().backward(); optim.step(); sched.step(); position+=1
    state={'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':sched.state_dict(),'rng':torch.get_rng_state(),'data_position':position}
    torch.save(state,path)
    print(digest(state))
if __name__=='__main__': run(Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3]=='resume')
