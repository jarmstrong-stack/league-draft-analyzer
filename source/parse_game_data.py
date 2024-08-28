"""
  Parses game data from `gol.gg` into a league-draft-analyzer readable format

  Json format:
  {
    "pick":{
      "blue": dict[int:Champ], # Where int describes champ's lane assignment
      "red":  dict[int:Champ]  # Where int describes champ's lane assignment 
    },

    "ban":{
      "blue": list[Champs],
      "red":  list[Champs]
    },

    "tournament": str,
    "game-time":  int,
    "teams":      dict[str1:str2], # Where str1 is blue/red and str2 is team name
    "game-date":  int
  }

  How to run:
  $ python3 source/parse_game_data.py <output.json>

  TODO: Make json format better for searches
"""
