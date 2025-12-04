#!/usr/bin/env python3
"""
Test Python bindings for Trigo C++ game engine
"""

import trigo_engine as tg

def test_basic_game():
	"""Test basic game operations"""
	print("=== Testing Python Bindings ===\n")
	
	# Create game
	print("1. Creating game with 5×5×5 board...")
	shape = tg.BoardShape(5, 5, 5)
	game = tg.TrigoGame(shape)
	print(f"   Board shape: {game.get_shape()}")
	print(f"   Current player: {game.get_current_player()}")
	
	# Start game
	game.start_game()
	print(f"   Game status: {game.get_game_status()}")
	
	# Make some moves
	print("\n2. Making moves...")
	pos1 = tg.Position(2, 2, 2)
	result1 = game.drop(pos1)
	print(f"   Black at {pos1}: {result1}")
	print(f"   Current player after: {game.get_current_player()}")
	
	pos2 = tg.Position(2, 2, 3)
	result2 = game.drop(pos2)
	print(f"   White at {pos2}: {result2}")
	
	# Check history
	print("\n3. Checking history...")
	history = game.get_history()
	print(f"   Moves made: {len(history)}")
	print(f"   Current step: {game.get_current_step()}")
	
	# Test undo
	print("\n4. Testing undo...")
	undo_result = game.undo()
	print(f"   Undo successful: {undo_result}")
	print(f"   Current step after undo: {game.get_current_step()}")
	print(f"   Stone at {pos2}: {game.get_stone(pos2)}")
	
	# Test redo
	print("\n5. Testing redo...")
	redo_result = game.redo()
	print(f"   Redo successful: {redo_result}")
	print(f"   Stone at {pos2} after redo: {game.get_stone(pos2)}")
	
	# Test territory
	print("\n6. Testing territory calculation...")
	territory = game.get_territory()
	print(f"   {territory}")
	
	# Test valid moves
	print("\n7. Testing valid move positions...")
	valid_positions = game.valid_move_positions()
	print(f"   Valid positions available: {len(valid_positions)}")
	
	# Test pass
	print("\n8. Testing pass...")
	game.pass_turn()
	print(f"   Pass count: {game.get_pass_count()}")
	
	print("\n=== All Tests Passed! ===")

if __name__ == '__main__':
	test_basic_game()
