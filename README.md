# Yahtzee-switch
This script computes the optimal action for the Yahtzee game (with simplified rules used in Nintendo switch)


Compute value function: python meta.py

For inference, use inference.py

## Explaination of the state vector: 

### Entry 0-11: Scoring Slots
  
  0-5: sum of numbers equal to i+1
  
  6: sum of dice numbers
  
  7: 4 of a kind
  
  8: full house
  
  9: small straight
  
  10: large straight
  
  11: yahtzee
  
### Entry 12: The current stage (0, 1, 2), each round starts with stage 0, which is increased by 1 every time you roll the dices. 

### Entry 13: The total score from the first 6 slots

### Entry 14-18: The current dice numbers, note that the number starts with 0, which represents dice number 1
