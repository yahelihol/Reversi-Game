

See image

# Othello/Reversi - Pygame

implementation of the  board game Othello/Reversi in Pygame
Check the internet for rules

Features:
- Classic 8x8 Othello board
- Legal move highlights and undo support.
- Turn tracking (black and white), Undo, Reset
- Scoreboard with win count

- Two-player support (human vs human, or human vs computer).
- Computer evaluation using alpha-beta pruning with increasing depth.
- Real-time best move and evaluation score display (uses another thread).


Game Controls:
Left-click on a highlighted legal move to place your piece.

Q	Hold to make random valid moves.
W	Undo last move.
R	Reset the game.
Close	Click window close button to exit.


See depth for how many moves ahead the (alpha-beta+pruning) algorithm sees
See score for the sum of pieces (positive is an advantage for black and negative for white)
Predicted score is for the supposed optimal play, and may show a guaranteed win

Yellow highlights the best move according to the computer