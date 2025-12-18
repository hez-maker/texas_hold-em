import random
from simulation import generate_deck,evaluate
deck =generate_deck()

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['♠', '♣', '♦', '♥']
rank1 = ['A', '2', '3', '4', '5']
suit1 = random.sample(suits, 1)
hand = [f'{rank}{suit1[0]}' for rank in rank1]

left_deck = [card for card in deck if card not in hand]
random.shuffle(left_deck)
hand += left_deck[:2]
random.shuffle(hand)

score, hand_value = evaluate(hand, [])


print(hand)
print(score, hand_value)