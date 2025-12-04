/**
 * TrigoGame - Main Game State Management
 *
 * C++ port of TypeScript TrigoGame class (inc/trigo/game.ts)
 * Integrates game state, move history, and game logic in a single class
 *
 * Key features:
 * - Maintains game board state
 * - Tracks complete move history
 * - Implements Go rules (capture, Ko, suicide)
 * - Supports undo/redo functionality
 * - Territory calculation
 */

#pragma once

#include "trigo_types.hpp"
#include "trigo_game_utils.hpp"
#include <vector>
#include <functional>
#include <optional>
#include <chrono>
#include <memory>


namespace trigo
{


/**
 * Step Types - Different types of moves in the game
 * Equivalent to StepType enum in TypeScript
 */
enum class StepType
{
	DROP = 0,      // Place a stone
	PASS = 1,      // Pass turn
	SURRENDER = 2, // Resign/surrender
	UNDO = 3       // Undo last move (called "REPENT" in prototype)
};


/**
 * Game Status enumeration
 */
enum class GameStatus
{
	IDLE,
	PLAYING,
	PAUSED,
	FINISHED
};


/**
 * Step - Represents a single move in the game
 * Equivalent to Step interface in TypeScript
 */
struct Step
{
	StepType type;
	std::optional<Position> position;  // Only for DROP moves
	Stone player;                      // Which player made this move
	std::vector<Position> capturedPositions; // Stones captured by this move
	std::chrono::system_clock::time_point timestamp; // When the move was made

	Step(StepType t, Stone p)
		: type(t), player(p), timestamp(std::chrono::system_clock::now()) {}

	Step(StepType t, Position pos, Stone p)
		: type(t), position(pos), player(p), timestamp(std::chrono::system_clock::now()) {}
};


/**
 * Game Callbacks - Event handlers for game state changes
 * Equivalent to GameCallbacks interface in TypeScript
 */
struct GameCallbacks
{
	std::function<void(const Step&, const std::vector<Step>&)> onStepAdvance;
	std::function<void(const Step&, const std::vector<Step>&)> onStepBack;
	std::function<void(const std::vector<Position>&)> onCapture;
	std::function<void(Stone)> onWin;
	std::function<void(const TerritoryResult&)> onTerritoryChange;
};


/**
 * TrigoGame - Main game class managing state, history, and logic
 *
 * Equivalent to TrigoGame class in TypeScript (lines 89-1211)
 */
class TrigoGame
{
public:
	// Constructor
	TrigoGame(
		const BoardShape& shape = BoardShape{5, 5, 5},
		const GameCallbacks& callbacks = GameCallbacks{}
	);

	// Destructor
	~TrigoGame() = default;

	// Copy constructor and assignment
	TrigoGame(const TrigoGame& other);
	TrigoGame& operator=(const TrigoGame& other);

	// Move constructor and assignment
	TrigoGame(TrigoGame&& other) noexcept = default;
	TrigoGame& operator=(TrigoGame&& other) noexcept = default;


	// === Game State Management ===

	/**
	 * Reset the game to initial state
	 * Equivalent to Game.reset() in TypeScript
	 */
	void reset();

	/**
	 * Clone the game state (deep copy)
	 * Creates an independent copy with all state preserved
	 */
	TrigoGame clone() const;


	// === Board Access ===

	/**
	 * Get current board state (read-only copy)
	 */
	Board get_board() const;

	/**
	 * Get stone at specific position
	 * Equivalent to Game.stone() in TypeScript
	 */
	Stone get_stone(const Position& pos) const;

	/**
	 * Get current player
	 */
	Stone get_current_player() const;

	/**
	 * Get current step number
	 * Equivalent to Game.currentStep() in TypeScript
	 */
	int get_current_step() const;

	/**
	 * Get move history
	 * Equivalent to Game.routine() in TypeScript
	 */
	std::vector<Step> get_history() const;

	/**
	 * Get last move
	 * Equivalent to Game.lastStep() in TypeScript
	 */
	std::optional<Step> get_last_step() const;

	/**
	 * Get board shape
	 * Equivalent to Game.shape() in TypeScript
	 */
	BoardShape get_shape() const;

	/**
	 * Get game status
	 */
	GameStatus get_game_status() const;

	/**
	 * Set game status
	 */
	void set_game_status(GameStatus status);

	/**
	 * Get game result
	 */
	std::optional<GameResult> get_game_result() const;

	/**
	 * Get consecutive pass count
	 */
	int get_pass_count() const;


	// === Game Actions ===

	/**
	 * Start the game
	 */
	void start_game();

	/**
	 * Check if game is active
	 */
	bool is_game_active() const;

	/**
	 * Check if a move is valid
	 * Equivalent to Game.isDropable() and Game.isValidStep() in TypeScript
	 */
	MoveValidation is_valid_move(const Position& pos, std::optional<Stone> player = std::nullopt) const;

	/**
	 * Get all valid move positions for current player (efficient batch query)
	 *
	 * This method is optimized to avoid repeated validation checks by:
	 * 1. Only checking empty positions
	 * 2. Skipping bounds checking (iterator is already within bounds)
	 * 3. Using low-level validation functions directly
	 * 4. Batching board state access
	 *
	 * @param player - Optional player color (defaults to current player)
	 * @returns Vector of all valid move positions
	 */
	std::vector<Position> valid_move_positions(std::optional<Stone> player = std::nullopt) const;

	/**
	 * Place a stone (drop move)
	 * Equivalent to Game.drop() and Game.appendStone() in TypeScript
	 *
	 * @returns true if move was successful, false otherwise
	 */
	bool drop(const Position& pos);

	/**
	 * Pass turn
	 * Equivalent to PASS step type in TypeScript
	 */
	bool pass();

	/**
	 * Surrender/resign
	 * Equivalent to Game.step() with SURRENDER type in TypeScript
	 */
	bool surrender();

	/**
	 * Undo last move
	 * Equivalent to Game.repent() in TypeScript
	 *
	 * @returns true if undo was successful, false if no moves to undo
	 */
	bool undo();

	/**
	 * Redo next move (after undo)
	 *
	 * @returns true if redo was successful, false if no moves to redo
	 */
	bool redo();

	/**
	 * Check if redo is available
	 */
	bool can_redo() const;

	/**
	 * Jump to specific step in history
	 * Rebuilds board state after applying the first 'index' moves
	 *
	 * @param index Number of moves to apply from history (0 for initial state, 1 for after first move, etc.)
	 * @returns true if jump was successful
	 */
	bool jump_to_step(int index);


	// === Territory and Statistics ===

	/**
	 * Get territory calculation
	 * Equivalent to Game.blackDomain() and Game.whiteDomain() in TypeScript
	 *
	 * Returns cached result if territory hasn't changed
	 */
	TerritoryResult get_territory();

	/**
	 * Get captured stone counts up to current position in history
	 * Only counts captures that have been played (up to currentStepIndex)
	 */
	struct CapturedCounts
	{
		int black;
		int white;
	};
	CapturedCounts get_captured_counts() const;

	/**
	 * Get game statistics
	 */
	struct GameStats
	{
		int totalMoves;
		int blackMoves;
		int whiteMoves;
		int capturedByBlack;
		int capturedByWhite;
		TerritoryResult territory;
	};
	GameStats get_stats();


private:
	// === Private Helper Methods ===

	/**
	 * Create an empty board
	 */
	Board create_empty_board() const;

	/**
	 * Recalculate consecutive pass count based on current history
	 * Counts consecutive PASS steps from the end of current history
	 */
	void recalculate_pass_count();

	/**
	 * Reset pass count (called when a stone is placed)
	 */
	void reset_pass_count();

	/**
	 * Advance to next step
	 * Equivalent to Game.stepAdvance() in TypeScript
	 */
	void advance_step(const Step& step);

	/**
	 * Deep copy the board
	 */
	Board deep_copy_board(const Board& board) const;

	/**
	 * Deep copy position vector
	 */
	std::vector<Position> deep_copy_positions(const std::vector<Position>& positions) const;


	// === Member Variables ===

	// Game configuration
	BoardShape shape;
	GameCallbacks callbacks;

	// Game state
	Board board;
	Stone currentPlayer;
	std::vector<Step> stepHistory;
	int currentStepIndex;

	// Game status management
	GameStatus gameStatus;
	std::optional<GameResult> gameResult;
	int passCount;

	// Last captured stones for Ko rule detection
	std::vector<Position> lastCapturedPositions;

	// Territory cache
	bool territoryDirty;
	std::optional<TerritoryResult> cachedTerritory;
};


} // namespace trigo
