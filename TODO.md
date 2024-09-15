# Synergy team values(win-rate based)
Compute champion pairings synergy win-rate based

# Make a "meta-proof" model
Find a way to make this model to be meta-proof in a way.

Maybe will need to re-train the model periodically with the latest games and give it some sort of priority?

# Parse champions into ML readable data
Make a sort of champion database, so we can parse champion like "Yone" to ML readable champion(maybe a number).

Maybe store a dinamically handled yml file to assign integer values to string champ's names.
(each time we find a new champ we add him to the yml file and give him the next incremental int value)

## Make utlity "search" functions for processed data
Make utility functions to search data in processed data(search teams like T1 or champs like Yone), calculate win-rate, etc..
