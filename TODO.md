# Improve:
1. Feature Engineering:
- Contextual Synergy: Instead of only using win-rate-based synergy values, you could introduce role-specific synergy interactions (e.g., how specific pairings like Top and Jungle perform together in general).
- Meta Awareness: Include a feature that reflects how certain champions perform based on the current patch and the overall win rate in the meta.
- Champion Class Info: Add champion types (e.g., tank, assassin, mage) as additional categorical features, giving the model more contextual information about roles.
- Patch Trends: Add the win rate or pick rate of each champion for a given patch as additional data.
- Champion Ban Synergies: Instead of just embedding bans, calculate the synergy loss caused by bans, reflecting how the strategy changes due to opponent bans.

2. Architectural Improvements:
- Attention on Synergies: You can apply attention layers to synergy inputs to focus on key pairings or role pairings that might impact the outcome more strongly.
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

# Expand network while checking what works and what doesnt
obviously weight decay doesnt do shit(xdd)

# feature engineer
expand synergy data...
lane specific picks & bans...

# Make a "meta-proof" model
Find a way to make this model to be meta-proof in a way.

Maybe will need to re-train the model periodically with the latest games and give it some sort of priority?

## Make utlity "search" functions for processed data
Make utility functions to search data in processed data(search teams like T1 or champs like Yone), calculate win-rate, etc..
