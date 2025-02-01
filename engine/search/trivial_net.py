import chess
import math

class TrivialNet:
    def evaluate(self, board : chess.Board):
        """
        Evaluate the board from the perspective of the side to move.
        """
        result = None
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)

        if result != None:
            if result == "1/2-1/2":
                return dict(), 0.0
            else:
                # Always return -1.0 when checkmated
                # and we are checkmated because it's our turn to move
                return dict(), -1.0
        
        policy = self.evaluate_policy(board)
        value = self.evaluate_value(board)
        
        return policy, value
    
    def evaluate_policy(self, board : chess.Board):
        """
        Evaluate the policy of a position from the perspective of the side to move.

        Gives a trivial policy of 1/n for each legal move, where n is the number of legal moves.
        """
        legal_moves = list(board.legal_moves)
        return {move.uci(): 1 / len(legal_moves) for move in legal_moves}


    def evaluate_value(self, board : chess.Board):
        """
        Evaluate the expected value of a position from the perspective of the side to move.
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King not counted in material
        }

        # Calculate material difference
        white_material = sum(len(board.pieces(piece_type, chess.WHITE)) * value 
                           for piece_type, value in piece_values.items())
        black_material = sum(len(board.pieces(piece_type, chess.BLACK)) * value 
                           for piece_type, value in piece_values.items())
        
        # Calculate relative score from white's perspective
        material_diff = white_material - black_material
        
        # Flip material_diff if it's black to move
        if not board.turn:  # board.turn is False for black
            material_diff = -material_diff

        # Normalize score to [-1, 1] using tanh
        # Division by 20 to get reasonable scaling (full queen = 9 points)
        score = math.tanh(material_diff / 20)
        
        return score
