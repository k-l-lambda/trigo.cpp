#!/bin/bash
#
# MCTS Performance Benchmark Script
# Compares C++ AlphaZero MCTS vs TypeScript MCTS performance
#
# Usage:
#   ./benchmark_mcts.sh [options]
#
# Options:
#   --board SHAPE         Board shape (default: 5x5x1)
#   --games N             Number of games (default: 10)
#   --simulations N       MCTS simulations per move (default: 50)
#   --cpp-model PATH      C++ model path (default: ../models/trained_shared)
#   --ts-dir PATH         TypeScript project directory
#   --output-dir PATH     Output directory (default: /tmp/mcts_benchmark)
#   --use-gpu             Enable GPU acceleration (default: CPU only)
#   --help                Show this help

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
BOARD_SHAPE="5x5x1"
NUM_GAMES=10
NUM_SIMULATIONS=50
CPP_MODEL_PATH="$PROJECT_ROOT/models/trained_shared"
TS_PROJECT_DIR="/home/camus/work/trigo/trigo-web"
OUTPUT_DIR="/tmp/mcts_benchmark_$(date +%Y%m%d_%H%M%S)"
CPP_BUILD_DIR="$PROJECT_ROOT/build"
USE_GPU=0  # 0 = CPU only (default), 1 = try GPU

# Parse command line arguments
while [[ $# -gt 0 ]]; do
	case $1 in
		--board)
			BOARD_SHAPE="$2"
			shift 2
			;;
		--games)
			NUM_GAMES="$2"
			shift 2
			;;
		--simulations)
			NUM_SIMULATIONS="$2"
			shift 2
			;;
		--cpp-model)
			CPP_MODEL_PATH="$2"
			shift 2
			;;
		--ts-dir)
			TS_PROJECT_DIR="$2"
			shift 2
			;;
		--output-dir)
			OUTPUT_DIR="$2"
			shift 2
			;;
		--use-gpu)
			USE_GPU=1
			shift
			;;
		--help)
			head -n 20 "$0" | grep "^#" | sed 's/^# //'
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information"
			exit 1
			;;
	esac
done


# Create output directories
mkdir -p "$OUTPUT_DIR/cpp_output"
mkdir -p "$OUTPUT_DIR/ts_output"

# Convert board shape for TypeScript format (5x5x1 -> 5*5*1)
TS_BOARD_SHAPE="${BOARD_SHAPE//x/*}"

# Validate C++ model path
if [ ! -f "$CPP_MODEL_PATH/base_model.onnx" ]; then
	echo "Error: C++ model not found at $CPP_MODEL_PATH"
	echo "Expected files:"
	echo "  - $CPP_MODEL_PATH/base_model.onnx"
	echo "  - $CPP_MODEL_PATH/policy_head.onnx"
	echo "  - $CPP_MODEL_PATH/value_head.onnx"
	echo ""
	echo "Please use --cpp-model to specify correct model path"
	exit 1
fi

# Validate C++ build directory
if [ ! -f "$CPP_BUILD_DIR/self_play_generator" ]; then
	echo "Error: C++ self_play_generator not found at $CPP_BUILD_DIR"
	echo "Please build the project first: cd $PROJECT_ROOT/build && cmake .. && make"
	exit 1
fi

# Validate TypeScript project directory
if [ ! -f "$TS_PROJECT_DIR/tools/selfPlayGames.ts" ]; then
	echo "Error: TypeScript project not found at $TS_PROJECT_DIR"
	echo "Please use --ts-dir to specify correct project path"
	exit 1
fi

echo "============================================================"
echo "MCTS Performance Benchmark"
echo "============================================================"
echo "Configuration:"
echo "  Board Shape:        $BOARD_SHAPE"
echo "  Number of Games:    $NUM_GAMES"
echo "  MCTS Simulations:   $NUM_SIMULATIONS"
echo "  Output Directory:   $OUTPUT_DIR"
echo "  GPU Acceleration:   $([ $USE_GPU -eq 1 ] && echo 'Enabled' || echo 'Disabled (CPU only)')"
echo ""
echo "C++ Configuration:"
echo "  Build Directory:    $CPP_BUILD_DIR"
echo "  Model Path:         $CPP_MODEL_PATH"
echo ""
echo "TypeScript Configuration:"
echo "  Project Directory:  $TS_PROJECT_DIR"
echo "  Board Format:       $TS_BOARD_SHAPE"
echo "============================================================"
echo ""


#
# Test 1: C++ AlphaZero MCTS
#
echo "[1/2] Running C++ AlphaZero MCTS..."
echo "--------------------------------------------------------------"

# Build command based on GPU setting
if [ $USE_GPU -eq 1 ]; then
	echo "Command: $CPP_BUILD_DIR/self_play_generator --num-games $NUM_GAMES --board $BOARD_SHAPE --black-policy alphazero --white-policy alphazero --model $CPP_MODEL_PATH --output $OUTPUT_DIR/cpp_output --seed 42"
	echo ""
	# Run C++ benchmark with GPU
	CPP_START=$(date +%s.%N)
	if $CPP_BUILD_DIR/self_play_generator \
		--num-games $NUM_GAMES \
		--board $BOARD_SHAPE \
		--black-policy alphazero \
		--white-policy alphazero \
		--model $CPP_MODEL_PATH \
		--output $OUTPUT_DIR/cpp_output \
		--seed 42 > "$OUTPUT_DIR/cpp_log.txt" 2>&1; then
		CPP_END=$(date +%s.%N)
		CPP_SUCCESS=1
		echo "✓ C++ test completed successfully"
	else
		CPP_END=$(date +%s.%N)
		CPP_SUCCESS=0
		echo "✗ C++ test failed (check $OUTPUT_DIR/cpp_log.txt)"
	fi
else
	echo "Command: TRIGO_FORCE_CPU=1 $CPP_BUILD_DIR/self_play_generator --num-games $NUM_GAMES --board $BOARD_SHAPE --black-policy alphazero --white-policy alphazero --model $CPP_MODEL_PATH --output $OUTPUT_DIR/cpp_output --seed 42"
	echo ""
	# Run C++ benchmark with CPU
	CPP_START=$(date +%s.%N)
	if TRIGO_FORCE_CPU=1 $CPP_BUILD_DIR/self_play_generator \
		--num-games $NUM_GAMES \
		--board $BOARD_SHAPE \
		--black-policy alphazero \
		--white-policy alphazero \
		--model $CPP_MODEL_PATH \
		--output $OUTPUT_DIR/cpp_output \
		--seed 42 > "$OUTPUT_DIR/cpp_log.txt" 2>&1; then
		CPP_END=$(date +%s.%N)
		CPP_SUCCESS=1
		echo "✓ C++ test completed successfully"
	else
		CPP_END=$(date +%s.%N)
		CPP_SUCCESS=0
		echo "✗ C++ test failed (check $OUTPUT_DIR/cpp_log.txt)"
	fi
fi

# Check for common GPU issues (only if GPU was attempted)
if [ $USE_GPU -eq 1 ] && [ $CPP_SUCCESS -eq 0 ]; then
	if grep -q "libcublasLt.so.12" "$OUTPUT_DIR/cpp_log.txt"; then
		echo ""
		echo "NOTE: GPU library version mismatch detected."
		echo "      System has CUDA 11.8, but ONNX Runtime 1.17.0 needs CUDA 12.x"
		echo "      Try running without --use-gpu flag to use CPU mode"
	fi
fi

CPP_DURATION=$(echo "$CPP_END - $CPP_START" | bc)
echo "Duration: ${CPP_DURATION}s"
echo ""


#
# Test 2: TypeScript MCTS
#
echo "[2/2] Running TypeScript MCTS..."
echo "--------------------------------------------------------------"

TS_CMD="cd $TS_PROJECT_DIR && \
	/home/camus/.nvm/versions/node/v21.7.1/bin/node \
	/home/camus/.nvm/versions/node/v21.7.1/bin/npx tsx \
	tools/selfPlayGames.ts \
	--games $NUM_GAMES \
	--board \"$TS_BOARD_SHAPE\" \
	--use-mcts \
	--mcts-simulations $NUM_SIMULATIONS \
	--output $OUTPUT_DIR/ts_output \
	--max-moves 200"

echo "Command: $TS_CMD"
echo ""

# Run TypeScript benchmark
TS_START=$(date +%s.%N)
if eval "$TS_CMD" > "$OUTPUT_DIR/ts_log.txt" 2>&1; then
	TS_END=$(date +%s.%N)
	TS_SUCCESS=1
	echo "✓ TypeScript test completed successfully"
else
	TS_END=$(date +%s.%N)
	TS_SUCCESS=0
	echo "✗ TypeScript test failed (check $OUTPUT_DIR/ts_log.txt)"
fi

TS_DURATION=$(echo "$TS_END - $TS_START" | bc)
echo "Duration: ${TS_DURATION}s"
echo ""


#
# Extract statistics from logs
#
echo "============================================================"
echo "Results Analysis"
echo "============================================================"

# Parse C++ results
if [ $CPP_SUCCESS -eq 1 ]; then
	CPP_GAMES=$(grep -oP "Generated \K\d+" "$OUTPUT_DIR/cpp_log.txt" | head -1 || echo "$NUM_GAMES")
	CPP_MOVES=$(grep -oP "Total moves: \K\d+" "$OUTPUT_DIR/cpp_log.txt" || echo "N/A")
	CPP_AVG_MOVES=$(grep -oP "Average game length: \K[\d.]+" "$OUTPUT_DIR/cpp_log.txt" || echo "N/A")

	if [ "$CPP_MOVES" != "N/A" ] && [ "$CPP_GAMES" != "0" ]; then
		CPP_AVG_MOVES=$(echo "scale=1; $CPP_MOVES / $CPP_GAMES" | bc)
	fi

	CPP_TIME_PER_GAME=$(echo "scale=2; $CPP_DURATION / $NUM_GAMES" | bc)

	# Calculate time per move
	if [ "$CPP_MOVES" != "N/A" ] && [ "$CPP_MOVES" != "0" ]; then
		CPP_TIME_PER_MOVE=$(echo "scale=0; ($CPP_DURATION * 1000) / $CPP_MOVES" | bc)
	else
		CPP_TIME_PER_MOVE="N/A"
	fi
else
	CPP_GAMES="FAILED"
	CPP_MOVES="N/A"
	CPP_AVG_MOVES="N/A"
	CPP_TIME_PER_GAME="N/A"
	CPP_TIME_PER_MOVE="N/A"
fi

# Parse TypeScript results
if [ $TS_SUCCESS -eq 1 ]; then
	TS_GAMES=$(grep -oP "Total games: \K\d+" "$OUTPUT_DIR/ts_log.txt" || echo "$NUM_GAMES")
	TS_MOVES=$(grep -oP "Total moves: \K\d+" "$OUTPUT_DIR/ts_log.txt" || echo "N/A")
	TS_AVG_MOVES=$(grep -oP "Average game length: \K[\d.]+" "$OUTPUT_DIR/ts_log.txt" || echo "N/A")

	if [ "$TS_AVG_MOVES" = "N/A" ] && [ "$TS_MOVES" != "N/A" ] && [ "$TS_GAMES" != "0" ]; then
		TS_AVG_MOVES=$(echo "scale=1; $TS_MOVES / $TS_GAMES" | bc)
	fi

	TS_TIME_PER_GAME=$(echo "scale=2; $TS_DURATION / $NUM_GAMES" | bc)

	# Calculate time per move
	if [ "$TS_MOVES" != "N/A" ] && [ "$TS_MOVES" != "0" ]; then
		TS_TIME_PER_MOVE=$(echo "scale=0; ($TS_DURATION * 1000) / $TS_MOVES" | bc)
	else
		TS_TIME_PER_MOVE="N/A"
	fi
else
	TS_GAMES="FAILED"
	TS_MOVES="N/A"
	TS_AVG_MOVES="N/A"
	TS_TIME_PER_GAME="N/A"
	TS_TIME_PER_MOVE="N/A"
fi

# Calculate speedup
if [ $CPP_SUCCESS -eq 1 ] && [ $TS_SUCCESS -eq 1 ]; then
	SPEEDUP=$(echo "scale=2; $TS_DURATION / $CPP_DURATION" | bc)
else
	SPEEDUP="N/A"
fi


#
# Print summary table
#
echo ""
echo "Performance Summary:"
echo "--------------------------------------------------------------"
printf "%-25s | %-15s | %-15s\n" "Metric" "C++ MCTS" "TypeScript MCTS"
echo "--------------------------------------------------------------"
printf "%-25s | %-15s | %-15s\n" "Total Duration" "${CPP_DURATION}s" "${TS_DURATION}s"
printf "%-25s | %-15s | %-15s\n" "Games Completed" "$CPP_GAMES" "$TS_GAMES"
printf "%-25s | %-15s | %-15s\n" "Total Moves" "$CPP_MOVES" "$TS_MOVES"
printf "%-25s | %-15s | %-15s\n" "Avg Moves per Game" "$CPP_AVG_MOVES" "$TS_AVG_MOVES"
printf "%-25s | %-15s | %-15s\n" "Time per Game" "${CPP_TIME_PER_GAME}s" "${TS_TIME_PER_GAME}s"
printf "%-25s | %-15s | %-15s\n" "Time per Move" "${CPP_TIME_PER_MOVE}ms" "${TS_TIME_PER_MOVE}ms"
echo "--------------------------------------------------------------"

if [ "$SPEEDUP" != "N/A" ]; then
	echo ""
	echo "Speedup: C++ is ${SPEEDUP}× faster than TypeScript"
	echo ""
fi


#
# Save report
#
REPORT_FILE="$OUTPUT_DIR/benchmark_report.txt"
cat > "$REPORT_FILE" <<EOF
MCTS Performance Benchmark Report
Generated: $(date)

Configuration:
  Board Shape:        $BOARD_SHAPE
  Number of Games:    $NUM_GAMES
  MCTS Simulations:   $NUM_SIMULATIONS
  C++ Model Path:     $CPP_MODEL_PATH
  TS Project Dir:     $TS_PROJECT_DIR

Results:
  C++ AlphaZero MCTS:
    Total Duration:     ${CPP_DURATION}s
    Games Completed:    $CPP_GAMES
    Total Moves:        $CPP_MOVES
    Avg Moves/Game:     $CPP_AVG_MOVES
    Time per Game:      ${CPP_TIME_PER_GAME}s
    Time per Move:      ${CPP_TIME_PER_MOVE}ms

  TypeScript MCTS:
    Total Duration:     ${TS_DURATION}s
    Games Completed:    $TS_GAMES
    Total Moves:        $TS_MOVES
    Avg Moves/Game:     $TS_AVG_MOVES
    Time per Game:      ${TS_TIME_PER_GAME}s
    Time per Move:      ${TS_TIME_PER_MOVE}ms

Performance:
  Speedup:              ${SPEEDUP}×

Output Files:
  C++ Log:              $OUTPUT_DIR/cpp_log.txt
  TypeScript Log:       $OUTPUT_DIR/ts_log.txt
  C++ Games:            $OUTPUT_DIR/cpp_output/*.tgn
  TypeScript Games:     $OUTPUT_DIR/ts_output/*.tgn
EOF

echo "Report saved to: $REPORT_FILE"
echo ""
echo "Log files:"
echo "  C++:        $OUTPUT_DIR/cpp_log.txt"
echo "  TypeScript: $OUTPUT_DIR/ts_log.txt"
echo ""
echo "Game files:"
echo "  C++:        $OUTPUT_DIR/cpp_output/"
echo "  TypeScript: $OUTPUT_DIR/ts_output/"
echo ""
echo "============================================================"
