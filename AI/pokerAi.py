from poker_ai import utils
from poker_ai.ai.dummy import RandomPlayer
from poker_ai.poker.table import PokerTable
from poker_ai.poker.engine import PokerEngine
from poker_ai.poker.pot import Pot


utils.random.seed(42)


initial_chips_amount = 10000
small_blind_amount = 50
big_blind_amount = 100


pot = Pot()

players = [
    RandomPlayer(
        name=f'player {player_i}',
        initial_chips=initial_chips_amount,
        pot=pot)
    for player_i in range(6)
]

table = PokerTable(players=players, pot=pot)

engine = PokerEngine(
    table=table,
    small_blind=small_blind_amount,
    big_blind=big_blind_amount)

engine.play_one_round()