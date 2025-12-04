/**
 * Python Bindings for Trigo C++ Game Engine
 *
 * Uses pybind11 to expose C++ TrigoGame class to Python
 * Enables integration with PyTorch/RL training frameworks
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../include/trigo_game.hpp"

namespace py = pybind11;
using namespace trigo;


/**
 * Python module: trigo_engine
 *
 * Provides Python interface to C++ Trigo game engine
 */
PYBIND11_MODULE(trigo_engine, m)
{
	m.doc() = "Trigo 3D Go game engine - C++ implementation with Python bindings";


	// === Enums ===

	py::enum_<Stone>(m, "Stone")
		.value("EMPTY", Stone::Empty)
		.value("BLACK", Stone::Black)
		.value("WHITE", Stone::White)
		.export_values();

	py::enum_<StepType>(m, "StepType")
		.value("DROP", StepType::DROP)
		.value("PASS", StepType::PASS)
		.value("SURRENDER", StepType::SURRENDER)
		.value("UNDO", StepType::UNDO)
		.export_values();

	py::enum_<GameStatus>(m, "GameStatus")
		.value("IDLE", GameStatus::IDLE)
		.value("PLAYING", GameStatus::PLAYING)
		.value("PAUSED", GameStatus::PAUSED)
		.value("FINISHED", GameStatus::FINISHED)
		.export_values();


	// === Structs ===

	py::class_<Position>(m, "Position")
		.def(py::init<>())
		.def(py::init<int, int, int>())
		.def_readwrite("x", &Position::x)
		.def_readwrite("y", &Position::y)
		.def_readwrite("z", &Position::z)
		.def("__repr__", [](const Position& p) {
			return "Position(" + std::to_string(p.x) + ", " +
			       std::to_string(p.y) + ", " +
			       std::to_string(p.z) + ")";
		})
		.def("__eq__", &Position::operator==)
		.def("__ne__", &Position::operator!=);

	py::class_<BoardShape>(m, "BoardShape")
		.def(py::init<>())
		.def(py::init<int, int, int>())
		.def_readwrite("x", &BoardShape::x)
		.def_readwrite("y", &BoardShape::y)
		.def_readwrite("z", &BoardShape::z)
		.def("__repr__", [](const BoardShape& s) {
			return "BoardShape(" + std::to_string(s.x) + ", " +
			       std::to_string(s.y) + ", " +
			       std::to_string(s.z) + ")";
		});

	py::class_<TerritoryResult>(m, "TerritoryResult")
		.def(py::init<>())
		.def_readwrite("black", &TerritoryResult::black)
		.def_readwrite("white", &TerritoryResult::white)
		.def_readwrite("neutral", &TerritoryResult::neutral)
		.def("__repr__", [](const TerritoryResult& t) {
			return "TerritoryResult(black=" + std::to_string(t.black) +
			       ", white=" + std::to_string(t.white) +
			       ", neutral=" + std::to_string(t.neutral) + ")";
		});

	py::class_<GameResult>(m, "GameResult")
		.def_readonly("winner", &GameResult::winner)
		.def_readonly("reason", &GameResult::reason);

	py::enum_<GameResult::Winner>(m, "GameResultWinner")
		.value("BLACK", GameResult::Winner::Black)
		.value("WHITE", GameResult::Winner::White)
		.value("DRAW", GameResult::Winner::Draw)
		.export_values();

	py::enum_<GameResult::Reason>(m, "GameResultReason")
		.value("RESIGNATION", GameResult::Reason::Resignation)
		.value("TIMEOUT", GameResult::Reason::Timeout)
		.value("COMPLETION", GameResult::Reason::Completion)
		.value("DOUBLE_PASS", GameResult::Reason::DoublePass)
		.export_values();

	py::class_<Step>(m, "Step")
		.def_readonly("type", &Step::type)
		.def_readonly("position", &Step::position)
		.def_readonly("player", &Step::player)
		.def_readonly("capturedPositions", &Step::capturedPositions);

	py::class_<MoveValidation>(m, "MoveValidation")
		.def_readonly("valid", &MoveValidation::valid)
		.def_readonly("reason", &MoveValidation::reason);


	// === Main TrigoGame Class ===

	py::class_<TrigoGame>(m, "TrigoGame")
		.def(py::init<>())
		.def(py::init<const BoardShape&>())

		// Game state management
		.def("reset", &TrigoGame::reset, "Reset game to initial state")
		.def("clone", &TrigoGame::clone, "Create deep copy of game")

		// Board access
		.def("get_board", &TrigoGame::get_board, "Get current board state (3D array)")
		.def("get_stone", &TrigoGame::get_stone, "Get stone at position")
		.def("get_current_player", &TrigoGame::get_current_player, "Get current player")
		.def("get_current_step", &TrigoGame::get_current_step, "Get current step number")
		.def("get_history", &TrigoGame::get_history, "Get move history")
		.def("get_last_step", &TrigoGame::get_last_step, "Get last move")
		.def("get_shape", &TrigoGame::get_shape, "Get board shape")
		.def("get_game_status", &TrigoGame::get_game_status, "Get game status")
		.def("set_game_status", &TrigoGame::set_game_status, "Set game status")
		.def("get_game_result", &TrigoGame::get_game_result, "Get game result")
		.def("get_pass_count", &TrigoGame::get_pass_count, "Get consecutive pass count")

		// Game actions
		.def("start_game", &TrigoGame::start_game, "Start the game")
		.def("is_game_active", &TrigoGame::is_game_active, "Check if game is active")
		.def("is_valid_move", &TrigoGame::is_valid_move,
		     "Check if move is valid",
		     py::arg("pos"), py::arg("player") = py::none())
		.def("valid_move_positions", &TrigoGame::valid_move_positions,
		     "Get all valid move positions",
		     py::arg("player") = py::none())
		.def("drop", &TrigoGame::drop, "Place a stone")
		.def("pass_turn", &TrigoGame::pass, "Pass turn")  // Renamed to avoid Python keyword
		.def("surrender", &TrigoGame::surrender, "Surrender/resign")
		.def("undo", &TrigoGame::undo, "Undo last move")
		.def("redo", &TrigoGame::redo, "Redo next move")
		.def("can_redo", &TrigoGame::can_redo, "Check if redo available")
		.def("jump_to_step", &TrigoGame::jump_to_step, "Jump to specific step")

		// Territory and statistics
		.def("get_territory", &TrigoGame::get_territory, "Calculate territory")
		.def("get_captured_counts", &TrigoGame::get_captured_counts, "Get captured stone counts")
		.def("get_stats", &TrigoGame::get_stats, "Get game statistics");


	// === Utility Functions ===

	m.def("create_board", &create_board, "Create empty board",
	      py::arg("shape"));

	m.def("get_neighbors", &get_neighbors, "Get neighboring positions",
	      py::arg("pos"), py::arg("shape"));

	m.def("is_in_bounds", &is_in_bounds, "Check if position is in bounds",
	      py::arg("pos"), py::arg("shape"));


	// === Module Metadata ===

	m.attr("__version__") = "1.0.0";
}
