/**
 * TrigoGame Implementation
 *
 * C++ port of TypeScript TrigoGame class (inc/trigo/game.ts)
 */

#include "../include/trigo_game.hpp"
#include <iostream>
#include <algorithm>


namespace trigo
{


// === Constructor ===

TrigoGame::TrigoGame(const BoardShape& shape, const GameCallbacks& callbacks)
	: shape(shape)
	, callbacks(callbacks)
	, board(create_empty_board())
	, currentPlayer(Stone::Black)
	, stepHistory()
	, currentStepIndex(0)
	, gameStatus(GameStatus::IDLE)
	, gameResult(std::nullopt)
	, passCount(0)
	, lastCapturedPositions()
	, territoryDirty(true)
	, cachedTerritory(std::nullopt)
{
}


// === Copy Constructor ===

TrigoGame::TrigoGame(const TrigoGame& other)
	: shape(other.shape)
	, callbacks(other.callbacks)
	, board(deep_copy_board(other.board))
	, currentPlayer(other.currentPlayer)
	, stepHistory(other.stepHistory)
	, currentStepIndex(other.currentStepIndex)
	, gameStatus(other.gameStatus)
	, gameResult(other.gameResult)
	, passCount(other.passCount)
	, lastCapturedPositions(other.lastCapturedPositions)
	, territoryDirty(true)
	, cachedTerritory(std::nullopt)
{
}


// === Assignment Operator ===

TrigoGame& TrigoGame::operator=(const TrigoGame& other)
{
	if (this != &other)
	{
		shape = other.shape;
		callbacks = other.callbacks;
		board = deep_copy_board(other.board);
		currentPlayer = other.currentPlayer;
		stepHistory = other.stepHistory;
		currentStepIndex = other.currentStepIndex;
		gameStatus = other.gameStatus;
		gameResult = other.gameResult;
		passCount = other.passCount;
		lastCapturedPositions = other.lastCapturedPositions;
		territoryDirty = true;
		cachedTerritory = std::nullopt;
	}
	return *this;
}


// === Game State Management ===

void TrigoGame::reset()
{
	board = create_empty_board();
	currentPlayer = Stone::Black;
	stepHistory.clear();
	currentStepIndex = 0;
	lastCapturedPositions.clear();
	territoryDirty = true;
	cachedTerritory = std::nullopt;
	gameStatus = GameStatus::IDLE;
	gameResult = std::nullopt;
	passCount = 0;
}


TrigoGame TrigoGame::clone() const
{
	return TrigoGame(*this);
}


// === Board Access ===

Board TrigoGame::get_board() const
{
	return deep_copy_board(board);
}


Stone TrigoGame::get_stone(const Position& pos) const
{
	return board[pos.x][pos.y][pos.z];
}


Stone TrigoGame::get_current_player() const
{
	return currentPlayer;
}


int TrigoGame::get_current_step() const
{
	return currentStepIndex;
}


std::vector<Step> TrigoGame::get_history() const
{
	return stepHistory;
}


std::optional<Step> TrigoGame::get_last_step() const
{
	if (currentStepIndex > 0)
	{
		return stepHistory[currentStepIndex - 1];
	}
	return std::nullopt;
}


BoardShape TrigoGame::get_shape() const
{
	return shape;
}


GameStatus TrigoGame::get_game_status() const
{
	return gameStatus;
}


void TrigoGame::set_game_status(GameStatus status)
{
	gameStatus = status;
}


std::optional<GameResult> TrigoGame::get_game_result() const
{
	return gameResult;
}


int TrigoGame::get_pass_count() const
{
	return passCount;
}


// === Game Actions ===

void TrigoGame::start_game()
{
	if (gameStatus == GameStatus::IDLE)
	{
		gameStatus = GameStatus::PLAYING;
	}
}


bool TrigoGame::is_game_active() const
{
	return gameStatus == GameStatus::PLAYING;
}


MoveValidation TrigoGame::is_valid_move(const Position& pos, std::optional<Stone> player) const
{
	Stone playerColor = player.value_or(currentPlayer);
	return validate_move(pos, playerColor, board, shape, &lastCapturedPositions);
}


std::vector<Position> TrigoGame::valid_move_positions(std::optional<Stone> player) const
{
	Stone playerColor = player.value_or(currentPlayer);
	std::vector<Position> validPositions;

	// Iterate through all board positions (bounds are guaranteed)
	for (int x = 0; x < shape.x; x++)
	{
		for (int y = 0; y < shape.y; y++)
		{
			for (int z = 0; z < shape.z; z++)
			{
				// Skip occupied positions (quick filter)
				if (board[x][y][z] != Stone::Empty)
				{
					continue;
				}

				Position pos(x, y, z);

				// Check Ko violation
				if (is_ko_violation(pos, playerColor, board, shape, &lastCapturedPositions))
				{
					continue;
				}

				// Check suicide rule
				if (is_suicide_move(pos, playerColor, board, shape))
				{
					continue;
				}

				// Position is valid
				validPositions.push_back(pos);
			}
		}
	}

	return validPositions;
}


bool TrigoGame::has_capturing_move(std::optional<Stone> player) const
{
	Stone playerColor = player.value_or(currentPlayer);

	// Iterate through all board positions
	for (int x = 0; x < shape.x; x++)
	{
		for (int y = 0; y < shape.y; y++)
		{
			for (int z = 0; z < shape.z; z++)
			{
				// Skip occupied positions
				if (board[x][y][z] != Stone::Empty)
				{
					continue;
				}

				Position pos(x, y, z);

				// Skip invalid moves (Ko, suicide)
				if (is_ko_violation(pos, playerColor, board, shape, &lastCapturedPositions))
				{
					continue;
				}

				if (is_suicide_move(pos, playerColor, board, shape))
				{
					continue;
				}

				// Check if this move would capture any stones
				auto capturedGroups = find_captured_groups(pos, playerColor, board, shape);
				if (!capturedGroups.empty())
				{
					return true;  // Found a capturing move
				}
			}
		}
	}

	return false;  // No capturing moves available
}


bool TrigoGame::has_any_capturing_move() const
{
	bool blackCanCapture = false;
	bool whiteCanCapture = false;

	// Single board traversal checking both players
	for (int x = 0; x < shape.x; x++)
	{
		for (int y = 0; y < shape.y; y++)
		{
			for (int z = 0; z < shape.z; z++)
			{
				// Skip occupied positions
				if (board[x][y][z] != Stone::Empty)
				{
					continue;
				}

				Position pos(x, y, z);

				// Check Black capturing move (if not already found)
				if (!blackCanCapture)
				{
					if (!is_ko_violation(pos, Stone::Black, board, shape, &lastCapturedPositions) &&
					    !is_suicide_move(pos, Stone::Black, board, shape))
					{
						auto capturedGroups = find_captured_groups(pos, Stone::Black, board, shape);
						if (!capturedGroups.empty())
						{
							blackCanCapture = true;
						}
					}
				}

				// Check White capturing move (if not already found)
				if (!whiteCanCapture)
				{
					if (!is_ko_violation(pos, Stone::White, board, shape, &lastCapturedPositions) &&
					    !is_suicide_move(pos, Stone::White, board, shape))
					{
						auto capturedGroups = find_captured_groups(pos, Stone::White, board, shape);
						if (!capturedGroups.empty())
						{
							whiteCanCapture = true;
						}
					}
				}

				// Early exit if both players have capturing moves
				if (blackCanCapture && whiteCanCapture)
				{
					return true;
				}
			}
		}
	}

	// Return true if at least one player has capturing move
	return (blackCanCapture || whiteCanCapture);
}


bool TrigoGame::drop(const Position& pos)
{
	// Validate the move
	auto validation = is_valid_move(pos);
	if (!validation.valid)
	{
		std::cerr << "Invalid move at (" << pos.x << ", " << pos.y << ", " << pos.z
		          << "): " << validation.reason << std::endl;
		return false;
	}

	// Find captured groups BEFORE placing the stone
	auto capturedGroups = find_captured_groups(pos, currentPlayer, board, shape);

	// Place the stone on the board
	set_stone(board, pos, currentPlayer);

	// Execute captures
	auto capturedPositions = execute_captures(capturedGroups, board);

	// Store captured positions for Ko rule
	lastCapturedPositions = capturedPositions;

	// Mark territory as dirty
	territoryDirty = true;

	// Reset pass count when a stone is placed
	reset_pass_count();

	// Create step record
	Step step(StepType::DROP, pos, currentPlayer);
	step.capturedPositions = capturedPositions;

	// Advance to next step
	advance_step(step);

	// Trigger callbacks
	if (!capturedPositions.empty() && callbacks.onCapture)
	{
		callbacks.onCapture(capturedPositions);
	}

	if (territoryDirty && callbacks.onTerritoryChange)
	{
		callbacks.onTerritoryChange(get_territory());
	}

	return true;
}


bool TrigoGame::pass()
{
	Step step(StepType::PASS, currentPlayer);

	lastCapturedPositions.clear();

	// Increment pass count
	passCount++;

	// Advance step
	advance_step(step);

	// Check for double pass (game end condition)
	if (passCount >= 2)
	{
		// Calculate territory to determine winner
		auto territory = get_territory();
		auto capturedCounts = get_captured_counts();
		int blackTotal = territory.black + capturedCounts.white; // black's territory + white stones captured
		int whiteTotal = territory.white + capturedCounts.black; // white's territory + black stones captured

		GameResult::Winner winner;
		if (blackTotal > whiteTotal)
		{
			winner = GameResult::Winner::Black;
		}
		else if (whiteTotal > blackTotal)
		{
			winner = GameResult::Winner::White;
		}
		else
		{
			winner = GameResult::Winner::Draw;
		}

		gameResult = GameResult(winner, GameResult::Reason::DoublePass);
		gameStatus = GameStatus::FINISHED;

		// Trigger win callback
		if (callbacks.onWin)
		{
			Stone winnerStone = winner == GameResult::Winner::Black ? Stone::Black
			                    : winner == GameResult::Winner::White ? Stone::White
			                    : Stone::Empty;
			callbacks.onWin(winnerStone);
		}
	}

	return true;
}


bool TrigoGame::surrender()
{
	Stone surrenderingPlayer = currentPlayer; // Remember who surrendered

	Step step(StepType::SURRENDER, currentPlayer);

	advance_step(step);

	// Set game result - opponent of surrendering player wins
	GameResult::Winner winner = surrenderingPlayer == Stone::Black
	                            ? GameResult::Winner::White
	                            : GameResult::Winner::Black;

	gameResult = GameResult(winner, GameResult::Reason::Resignation);
	gameStatus = GameStatus::FINISHED;

	// Trigger win callback for the opponent
	Stone winnerStone = opponent_stone(surrenderingPlayer);
	if (callbacks.onWin)
	{
		callbacks.onWin(winnerStone);
	}

	return true;
}


bool TrigoGame::undo()
{
	if (currentStepIndex == 0 || stepHistory.empty())
	{
		return false;
	}

	const Step& lastStep = stepHistory[currentStepIndex - 1];

	// Revert the move
	if (lastStep.type == StepType::DROP && lastStep.position)
	{
		// Remove the placed stone
		set_stone(board, *lastStep.position, Stone::Empty);

		// Restore captured stones
		if (!lastStep.capturedPositions.empty())
		{
			Stone enemyColor = opponent_stone(lastStep.player);
			for (const auto& pos : lastStep.capturedPositions)
			{
				set_stone(board, pos, enemyColor);
			}
		}
	}

	// Move back in history
	currentStepIndex--;
	currentPlayer = lastStep.player; // Restore player who made that move

	// Recalculate pass count based on new history position
	recalculate_pass_count();

	// Update last captured positions for Ko rule
	// Need to check the step before this one
	if (currentStepIndex > 0)
	{
		const Step& previousStep = stepHistory[currentStepIndex - 1];
		lastCapturedPositions = previousStep.capturedPositions;
	}
	else
	{
		lastCapturedPositions.clear();
	}

	// Mark territory as dirty
	territoryDirty = true;

	// Trigger callback
	if (callbacks.onStepBack)
	{
		std::vector<Step> currentHistory(stepHistory.begin(), stepHistory.begin() + currentStepIndex);
		callbacks.onStepBack(lastStep, currentHistory);
	}

	return true;
}


bool TrigoGame::redo()
{
	// Check if we can redo (not at the end of history)
	if (currentStepIndex >= static_cast<int>(stepHistory.size()))
	{
		return false;
	}

	const Step& nextStep = stepHistory[currentStepIndex];

	// Re-apply the move
	if (nextStep.type == StepType::DROP && nextStep.position)
	{
		// Place the stone
		set_stone(board, *nextStep.position, nextStep.player);

		// Re-execute captures if there were any
		if (!nextStep.capturedPositions.empty())
		{
			for (const auto& pos : nextStep.capturedPositions)
			{
				set_stone(board, pos, Stone::Empty);
			}
		}

		// Update last captured positions
		lastCapturedPositions = nextStep.capturedPositions;
	}
	else if (nextStep.type == StepType::PASS)
	{
		lastCapturedPositions.clear();
	}

	// Move forward in history
	currentStepIndex++;
	currentPlayer = opponent_stone(nextStep.player); // Switch to next player

	// Mark territory as dirty
	territoryDirty = true;

	// Trigger callback
	if (callbacks.onStepAdvance)
	{
		std::vector<Step> currentHistory(stepHistory.begin(), stepHistory.begin() + currentStepIndex);
		callbacks.onStepAdvance(nextStep, currentHistory);
	}

	return true;
}


bool TrigoGame::can_redo() const
{
	return currentStepIndex < static_cast<int>(stepHistory.size());
}


bool TrigoGame::jump_to_step(int index)
{
	// Validate index: allow 0 (initial state) up to stepHistory.size() (all moves applied)
	if (index < 0 || index > static_cast<int>(stepHistory.size()))
	{
		return false;
	}

	// If already at target index, return false (no change made)
	if (index == currentStepIndex)
	{
		return false;
	}

	// Rebuild board from scratch
	board = create_empty_board();
	lastCapturedPositions.clear();

	// Replay all moves up to (but not including) target index
	// After this loop, we'll have applied 'index' number of moves
	for (int i = 0; i < index; i++)
	{
		const Step& step = stepHistory[i];

		if (step.type == StepType::DROP && step.position)
		{
			const Position& pos = *step.position;

			// Place the stone
			set_stone(board, pos, step.player);

			// Re-execute captures
			if (!step.capturedPositions.empty())
			{
				for (const auto& capturedPos : step.capturedPositions)
				{
					set_stone(board, capturedPos, Stone::Empty);
				}
			}
		}
	}

	// Set last captured positions from the last applied move (if any)
	if (index > 0)
	{
		const Step& lastAppliedStep = stepHistory[index - 1];
		if (lastAppliedStep.type == StepType::DROP)
		{
			lastCapturedPositions = lastAppliedStep.capturedPositions;
		}
		else if (lastAppliedStep.type == StepType::PASS)
		{
			lastCapturedPositions.clear();
		}
	}
	else
	{
		lastCapturedPositions.clear();
	}

	// Update current index
	int oldStepIndex = currentStepIndex;
	currentStepIndex = index;

	// Set current player based on number of moves played
	// currentStepIndex represents the number of moves applied
	int movesPlayed = index;
	currentPlayer = movesPlayed % 2 == 0 ? Stone::Black : Stone::White;

	// Recalculate pass count based on new history position
	recalculate_pass_count();

	// Mark territory as dirty
	territoryDirty = true;

	// Trigger callback based on direction
	if (index < oldStepIndex && callbacks.onStepBack)
	{
		std::vector<Step> currentHistory(stepHistory.begin(), stepHistory.begin() + index + 1);
		if (index < static_cast<int>(stepHistory.size()))
		{
			const Step& currentStep = stepHistory[index];
			callbacks.onStepBack(currentStep, currentHistory);
		}
	}
	else if (index > oldStepIndex && callbacks.onStepAdvance)
	{
		std::vector<Step> currentHistory(stepHistory.begin(), stepHistory.begin() + index + 1);
		if (index < static_cast<int>(stepHistory.size()))
		{
			const Step& currentStep = stepHistory[index];
			callbacks.onStepAdvance(currentStep, currentHistory);
		}
	}

	return true;
}


// === Territory and Statistics ===

TerritoryResult TrigoGame::get_territory()
{
	if (territoryDirty || !cachedTerritory)
	{
		cachedTerritory = calculate_territory(board, shape);
		territoryDirty = false;
	}
	return *cachedTerritory;
}


TrigoGame::CapturedCounts TrigoGame::get_captured_counts() const
{
	CapturedCounts counts{0, 0};

	// Only count captures up to current step index
	for (int i = 0; i < currentStepIndex; i++)
	{
		const Step& step = stepHistory[i];
		if (!step.capturedPositions.empty())
		{
			// Captured stones belong to the enemy of the player who made the move
			Stone enemyColor = opponent_stone(step.player);
			if (enemyColor == Stone::Black)
			{
				counts.black += step.capturedPositions.size();
			}
			else if (enemyColor == Stone::White)
			{
				counts.white += step.capturedPositions.size();
			}
		}
	}

	return counts;
}


TrigoGame::GameStats TrigoGame::get_stats()
{
	auto captured = get_captured_counts();
	auto territory = get_territory();

	int blackMoves = 0;
	int whiteMoves = 0;

	for (int i = 0; i < currentStepIndex; i++)
	{
		const Step& step = stepHistory[i];
		if (step.type == StepType::DROP)
		{
			if (step.player == Stone::Black)
			{
				blackMoves++;
			}
			else if (step.player == Stone::White)
			{
				whiteMoves++;
			}
		}
	}

	return GameStats{
		currentStepIndex,
		blackMoves,
		whiteMoves,
		captured.white,  // Black captures white stones
		captured.black,  // White captures black stones
		territory
	};
}


// === Private Helper Methods ===

Board TrigoGame::create_empty_board() const
{
	return trigo::create_board(shape);
}


void TrigoGame::recalculate_pass_count()
{
	passCount = 0;

	// Count backwards from current position to find consecutive passes
	for (int i = currentStepIndex - 1; i >= 0; i--)
	{
		if (stepHistory[i].type == StepType::PASS)
		{
			passCount++;
		}
		else
		{
			break; // Stop at first non-pass move
		}
	}
}


void TrigoGame::reset_pass_count()
{
	passCount = 0;
}


void TrigoGame::advance_step(const Step& step)
{
	// If we're not at the end of history, truncate future steps
	if (currentStepIndex < static_cast<int>(stepHistory.size()))
	{
		stepHistory.erase(stepHistory.begin() + currentStepIndex, stepHistory.end());
	}

	// Add the new step
	stepHistory.push_back(step);
	currentStepIndex++;

	// Switch player
	currentPlayer = opponent_stone(currentPlayer);

	// Trigger callback
	if (callbacks.onStepAdvance)
	{
		callbacks.onStepAdvance(step, stepHistory);
	}
}


Board TrigoGame::deep_copy_board(const Board& source) const
{
	Board copy;
	copy.reserve(source.size());

	for (const auto& plane : source)
	{
		std::vector<std::vector<Stone>> copyPlane;
		copyPlane.reserve(plane.size());

		for (const auto& row : plane)
		{
			copyPlane.push_back(row);
		}

		copy.push_back(copyPlane);
	}

	return copy;
}


std::vector<Position> TrigoGame::deep_copy_positions(const std::vector<Position>& positions) const
{
	return positions; // Position is a simple struct, copy is automatic
}


} // namespace trigo
