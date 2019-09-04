# -*- coding: utf-8 -*-
import torch



#params to learn
# dmg_nor
# dmg_exp
# dmg_con
# small_armor
# small_hit
# medium_armor
# medium_hit
# large_armor
# large_hit


x = torch.tensor(1.0)
dmg_nor = torch.tensor([10.0], requires_grad=True)
dmg_exp = torch.tensor([10.0], requires_grad=True)
dmg_con = torch.tensor([10.0], requires_grad=True)
small_armor = torch.tensor([1.0], requires_grad=True)
small_hit = torch.tensor([200.0], requires_grad=True)
medium_armor = torch.tensor([1.0], requires_grad=True)
medium_hit = torch.tensor([200.0], requires_grad=True)
large_armor = torch.tensor([1.0], requires_grad=True)
large_hit = torch.tensor([200.0], requires_grad=True)
y = torch.tensor(0.03) * 100

learning_rate = 1
for t in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y1 = ((x*dmg_nor) - small_armor) / small_hit
    y2 = ((x*dmg_nor) - medium_armor) / medium_hit
    y3 = ((x*dmg_nor) - large_armor) / large_hit
    y4 = ((x*dmg_con) - small_armor) / small_hit
    y5 = ((x*dmg_con) - medium_armor) * 0.5 / small_hit
    y6 = ((x*dmg_con) - large_armor) * 0.25 / small_hit
    y7 = ((x*dmg_exp) - small_armor) * 0.5 / small_hit
    y8 = ((x*dmg_exp) - medium_armor) * 0.75 / small_hit
    y9 = ((x*dmg_exp) - large_armor) / small_hit
    y_pred = torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9)) * 100
    
    #print(y_pred)
    # Compute and print loss
    loss = (y_pred-y).pow(2).mul(torch.tensor([9.,6.,3.,6.,4.,2.,3.,2.,1.])).sum()
    print(t, loss.item())

    loss.backward()
    with torch.no_grad():
        dmg_nor -= learning_rate * dmg_nor.grad
        dmg_exp -= learning_rate * dmg_exp.grad
        dmg_con -= learning_rate * dmg_con.grad
        small_armor -= learning_rate * small_armor.grad
        small_hit -= learning_rate * small_hit.grad
        medium_armor -= learning_rate * medium_armor.grad
        medium_hit -= learning_rate * medium_hit.grad
        large_armor -= learning_rate * large_armor.grad
        large_hit -= learning_rate * large_hit.grad
        small_armor[0] = max(small_armor[0],0.0)
        medium_armor[0] = max(medium_armor[0],0.0)
        large_armor[0] = max(large_armor[0],0.0)
        small_armor[0] = min(small_armor[0],10.0)
        medium_armor[0] = min(medium_armor[0],10.0)
        large_armor[0] = min(large_armor[0],10.0)
        small_hit[0] = min(small_hit[0],1000.0)
        medium_hit[0] = min(medium_hit[0],1000.0)
        large_hit[0] = min(large_hit[0],1000.0)

        # Manually zero the gradients after updating weights
        dmg_nor.grad.zero_()
        dmg_exp.grad.zero_()
        dmg_con.grad.zero_()
        small_armor.grad.zero_()
        small_hit.grad.zero_()
        medium_armor.grad.zero_()
        medium_hit.grad.zero_()
        large_armor.grad.zero_()
        large_hit.grad.zero_()
print(dmg_nor.item(),        
        dmg_con.item(),
        dmg_exp.item(),
        small_armor.item(),
        small_hit.item(),
        medium_armor.item(),
        medium_hit.item(),
        large_armor.item(),
        large_hit.item()
        )
