# Improve:
1. Feature Engineering:
- Champion Class Info: Add champion types (e.g., tank, assassin, mage) as additional categorical features, giving the model more contextual information about roles.

2. Architectural Improvements:
- Attention on Synergies: You can apply attention layers to synergy inputs to focus on key pairings or role pairings that might impact the outcome more strongly.
Recheck this ^
- Multi-Headed Attention on Roles: Since you have role-based embeddings, you could apply separate attention heads for different role interactions (e.g., Top & Jungle synergy).
- Residual Connections: Consider adding residual connections between some fully connected layers to retain lower-level information throughout deeper layers, which could help avoid vanishing gradient issues.
- Layer Normalization: Implement layer normalization or batch normalization between layers to stabilize learning and potentially improve convergence speed.

3. Regularization & Hyperparameters:
- Learning Rate Decay: Try using a learning rate scheduler that reduces the learning rate over time.
- Ensemble Models: Train multiple models with different initializations or architectures, and combine them in an ensemble to improve accuracy.
- Dropout Tweaking: Adjust the dropout rate. Too much dropout can prevent the network from learning important patterns, while too little can lead to overfitting.

4. Loss Function Adjustments:
- Weighted Loss: If certain drafts are harder to predict or more frequent, adjust the loss function to weight mispredictions differently, focusing more on difficult-to-predict cases.

# README.md
model noticed that blue side has better odds
model understood each champ becase of embedding layers
model understood what champions to prioritize in draft because of attention layers
difference between **good draft** and **good pro play draft**

# Some ai prompts:
- Attention on synergies
I implemented attention on synergies but the model didn't get higher accuracy then when i had no attention on them.
Can you check if my implementation is correct, from what I understood I should input to the linear layer
the ammount of synergies i have and the Linear layer will output the same ammount of synergies.
And with the output of the layers i apply attention and get the respective attention values. Please let me know if this is the correct way to implement attention to synergies.
([define(), forward()])
PS: Try NOT passing them through the embedding(linear) layer and directly into attention

- Multi-Headed Attention:
You wrote this in an earlier response:
- Multi-Headed Attention on Roles: Since you have role-based embeddings, you could apply separate attention heads for different role interactions (e.g., Top & Jungle synergy).
Can you explain this better? how would you implement it and how does it work, and will it improve the accuracy of the network in any way?
([define(), forward()])

- Weighted loss:
You wrote this in an earlier response:
- Weighted Loss: If certain drafts are harder to predict or more frequent, adjust the loss function to weight mispredictions differently, focusing more on difficult-to-predict cases.
And it seems interesting for our model, because maybe when a draft is "better" than the other they can end up losing...
How would weighted loss work and how would it be implemented? Would it hurt the learning of the model?
([define(), forward()])

# Expand network while checking what works and what doesnt
obviously weight decay doesnt do shit(xdd), it does qxdd

### Make utlity "search" functions for processed data
Make utility functions to search data in processed data(search teams like T1 or champs like Yone), calculate win-rate, etc..
