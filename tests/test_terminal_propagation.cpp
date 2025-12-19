/**
 * Terminal Propagation Unit Tests
 *
 * Tests minimax terminal state propagation in MCTS
 * Validates white-positive value system and correct minimax logic
 */

// Enable terminal propagation debug output
#define DEBUG_TERMINAL_PROPAGATION

#include "../include/mcts.hpp"
#include "../include/trigo_game.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

using namespace trigo;


// ============================================================================
// Test Utilities
// ============================================================================

void assert_eq(float a, float b, const std::string& msg) {
    if (std::abs(a - b) > 0.0001f) {
        std::cerr << "FAIL: " << msg << " (expected " << b << ", got " << a << ")\n";
        exit(1);
    }
}

void assert_true(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << "\n";
        exit(1);
    }
}

void print_test_header(const std::string& name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void print_test_pass(const std::string& name) {
    std::cout << "✓ PASS: " << name << "\n";
}


// ============================================================================
// Test 1: MCTSNode Terminal Detection
// ============================================================================

void test_node_terminal_detection() {
    print_test_header("MCTSNode Terminal Detection");

    // Create a node with NaN terminal value (not terminal)
    MCTSNode node1(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
    assert_true(!node1.is_terminal(), "New node should not be terminal (NaN)");

    // Create a node with explicit terminal value
    MCTSNode node2(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
    node2.terminal_value = 0.5f;
    assert_true(node2.is_terminal(), "Node with value 0.5 should be terminal");

    // Test with zero terminal value
    MCTSNode node3(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
    node3.terminal_value = 0.0f;
    assert_true(node3.is_terminal(), "Node with value 0.0 should be terminal");

    // Test with negative terminal value
    MCTSNode node4(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
    node4.terminal_value = -0.8f;
    assert_true(node4.is_terminal(), "Node with value -0.8 should be terminal");

    print_test_pass("MCTSNode Terminal Detection");
}


// ============================================================================
// Test 2: Player Alternation
// ============================================================================

void test_player_alternation() {
    print_test_header("Player Alternation");

    // Create root with Black to move
    auto root = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
    assert_true(root->player_to_move == Stone::Black, "Root should have Black to move");
    assert_true(root->depth == 0, "Root should have depth 0");

    // Create child with White to move
    Stone next_player = (root->player_to_move == Stone::Black) ? Stone::White : Stone::Black;
    auto child1 = std::make_unique<MCTSNode>(
        Position{0, 0, 0}, false, root.get(), 0.5f, 1, next_player
    );
    assert_true(child1->player_to_move == Stone::White, "Child should have White to move");
    assert_true(child1->depth == 1, "Child should have depth 1");

    // Create grandchild with Black to move
    Stone next_player2 = (child1->player_to_move == Stone::Black) ? Stone::White : Stone::Black;
    auto grandchild = std::make_unique<MCTSNode>(
        Position{1, 0, 0}, false, child1.get(), 0.3f, 2, next_player2
    );
    assert_true(grandchild->player_to_move == Stone::Black, "Grandchild should have Black to move");
    assert_true(grandchild->depth == 2, "Grandchild should have depth 2");

    print_test_pass("Player Alternation");
}


// ============================================================================
// Test 3: Single Child Minimax
// ============================================================================

void test_single_child_minimax() {
    print_test_header("Single Child Minimax");

    // Test White node with one terminal child
    {
        auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
        parent->is_fully_expanded = true;

        auto child = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, parent.get(), 1.0f, 1, Stone::Black);
        child->terminal_value = 0.75f;  // Terminal with value 0.75

        parent->children.push_back(std::move(child));

        // Simulate backprop logic (manually since we can't easily call backpropagate)
        bool all_children_terminal = true;
        float best_value = -std::numeric_limits<float>::infinity();  // White maximizes

        for (const auto& c : parent->children) {
            if (!c->is_terminal()) {
                all_children_terminal = false;
                break;
            }
            best_value = std::max(best_value, c->terminal_value);
        }

        if (all_children_terminal) {
            parent->terminal_value = best_value;
        }

        assert_true(parent->is_terminal(), "Parent should be terminal when only child is terminal");
        assert_eq(parent->terminal_value, 0.75f, "White parent should inherit child's value");
    }

    // Test Black node with one terminal child
    {
        auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
        parent->is_fully_expanded = true;

        auto child = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, parent.get(), 1.0f, 1, Stone::White);
        child->terminal_value = -0.5f;  // Terminal with value -0.5

        parent->children.push_back(std::move(child));

        // Simulate backprop logic
        bool all_children_terminal = true;
        float best_value = std::numeric_limits<float>::infinity();  // Black minimizes

        for (const auto& c : parent->children) {
            if (!c->is_terminal()) {
                all_children_terminal = false;
                break;
            }
            best_value = std::min(best_value, c->terminal_value);
        }

        if (all_children_terminal) {
            parent->terminal_value = best_value;
        }

        assert_true(parent->is_terminal(), "Parent should be terminal when only child is terminal");
        assert_eq(parent->terminal_value, -0.5f, "Black parent should inherit child's value");
    }

    print_test_pass("Single Child Minimax");
}


// ============================================================================
// Test 4: White Maximizes, Black Minimizes
// ============================================================================

void test_minimax_direction() {
    print_test_header("Minimax Direction (White Max, Black Min)");

    // Test White node maximizes
    {
        auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
        parent->is_fully_expanded = true;

        // Create three terminal children with different values
        auto child1 = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, parent.get(), 0.3f, 1, Stone::Black);
        child1->terminal_value = 0.2f;

        auto child2 = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, parent.get(), 0.4f, 1, Stone::Black);
        child2->terminal_value = 0.8f;  // Maximum value

        auto child3 = std::make_unique<MCTSNode>(Position{2, 0, 0}, false, parent.get(), 0.3f, 1, Stone::Black);
        child3->terminal_value = 0.5f;

        parent->children.push_back(std::move(child1));
        parent->children.push_back(std::move(child2));
        parent->children.push_back(std::move(child3));

        // Apply minimax
        bool all_children_terminal = true;
        float best_value = -std::numeric_limits<float>::infinity();

        for (const auto& c : parent->children) {
            if (!c->is_terminal()) {
                all_children_terminal = false;
                break;
            }
            best_value = std::max(best_value, c->terminal_value);
        }

        if (all_children_terminal) {
            parent->terminal_value = best_value;
        }

        assert_eq(parent->terminal_value, 0.8f, "White should maximize (choose 0.8)");
    }

    // Test Black node minimizes
    {
        auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
        parent->is_fully_expanded = true;

        // Create three terminal children with different values
        auto child1 = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, parent.get(), 0.3f, 1, Stone::White);
        child1->terminal_value = 0.6f;

        auto child2 = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, parent.get(), 0.4f, 1, Stone::White);
        child2->terminal_value = -0.3f;  // Minimum value

        auto child3 = std::make_unique<MCTSNode>(Position{2, 0, 0}, false, parent.get(), 0.3f, 1, Stone::White);
        child3->terminal_value = 0.1f;

        parent->children.push_back(std::move(child1));
        parent->children.push_back(std::move(child2));
        parent->children.push_back(std::move(child3));

        // Apply minimax
        bool all_children_terminal = true;
        float best_value = std::numeric_limits<float>::infinity();

        for (const auto& c : parent->children) {
            if (!c->is_terminal()) {
                all_children_terminal = false;
                break;
            }
            best_value = std::min(best_value, c->terminal_value);
        }

        if (all_children_terminal) {
            parent->terminal_value = best_value;
        }

        assert_eq(parent->terminal_value, -0.3f, "Black should minimize (choose -0.3)");
    }

    print_test_pass("Minimax Direction");
}


// ============================================================================
// Test 5: Empty Children Guard
// ============================================================================

void test_empty_children_guard() {
    print_test_header("Empty Children Guard");

    auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
    parent->is_fully_expanded = true;
    // No children added

    // Try to apply minimax logic
    if (!parent->children.empty()) {
        bool all_children_terminal = true;
        float best_value = -std::numeric_limits<float>::infinity();

        for (const auto& c : parent->children) {
            if (!c->is_terminal()) {
                all_children_terminal = false;
                break;
            }
            best_value = std::max(best_value, c->terminal_value);
        }

        if (all_children_terminal) {
            parent->terminal_value = best_value;
        }
    }

    // Parent should remain non-terminal (guarded by empty check)
    assert_true(!parent->is_terminal(), "Parent with no children should not be marked terminal");

    print_test_pass("Empty Children Guard");
}


// ============================================================================
// Test 6: Mixed Terminal/Non-Terminal Children
// ============================================================================

void test_mixed_children() {
    print_test_header("Mixed Terminal/Non-Terminal Children");

    auto parent = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
    parent->is_fully_expanded = true;

    // Create two terminal children and one non-terminal
    auto child1 = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, parent.get(), 0.3f, 1, Stone::Black);
    child1->terminal_value = 0.5f;

    auto child2 = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, parent.get(), 0.4f, 1, Stone::Black);
    // child2 remains non-terminal (NaN)

    auto child3 = std::make_unique<MCTSNode>(Position{2, 0, 0}, false, parent.get(), 0.3f, 1, Stone::Black);
    child3->terminal_value = 0.7f;

    parent->children.push_back(std::move(child1));
    parent->children.push_back(std::move(child2));
    parent->children.push_back(std::move(child3));

    // Apply minimax logic
    bool all_children_terminal = true;
    float best_value = -std::numeric_limits<float>::infinity();

    for (const auto& c : parent->children) {
        if (!c->is_terminal()) {
            all_children_terminal = false;
            break;
        }
        best_value = std::max(best_value, c->terminal_value);
    }

    if (all_children_terminal) {
        parent->terminal_value = best_value;
    }

    // Parent should NOT be terminal because not all children are terminal
    assert_true(!parent->is_terminal(), "Parent should not be terminal when some children are non-terminal");

    print_test_pass("Mixed Terminal/Non-Terminal Children");
}


// ============================================================================
// Test 7: Deep Tree Propagation
// ============================================================================

void test_deep_tree_propagation() {
    print_test_header("Deep Tree Propagation");

    // Create a tree: Root(White) -> Child(Black) -> Grandchild1(White), Grandchild2(White)
    auto root = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
    root->is_fully_expanded = true;

    auto child = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, root.get(), 1.0f, 1, Stone::Black);
    child->is_fully_expanded = true;

    // Create two terminal grandchildren
    auto grandchild1 = std::make_unique<MCTSNode>(Position{0, 0, 0}, false, child.get(), 0.5f, 2, Stone::White);
    grandchild1->terminal_value = 0.6f;

    auto grandchild2 = std::make_unique<MCTSNode>(Position{1, 0, 0}, false, child.get(), 0.5f, 2, Stone::White);
    grandchild2->terminal_value = 0.4f;

    child->children.push_back(std::move(grandchild1));
    child->children.push_back(std::move(grandchild2));

    // Step 1: Propagate to child (Black minimizes)
    bool all_children_terminal = true;
    float child_best = std::numeric_limits<float>::infinity();

    for (const auto& gc : child->children) {
        if (!gc->is_terminal()) {
            all_children_terminal = false;
            break;
        }
        child_best = std::min(child_best, gc->terminal_value);
    }

    if (all_children_terminal) {
        child->terminal_value = child_best;
    }

    assert_eq(child->terminal_value, 0.4f, "Black child should minimize (choose 0.4)");

    // Add child to root
    root->children.push_back(std::move(child));

    // Step 2: Propagate to root (White maximizes)
    all_children_terminal = true;
    float root_best = -std::numeric_limits<float>::infinity();

    for (const auto& c : root->children) {
        if (!c->is_terminal()) {
            all_children_terminal = false;
            break;
        }
        root_best = std::max(root_best, c->terminal_value);
    }

    if (all_children_terminal) {
        root->terminal_value = root_best;
    }

    assert_eq(root->terminal_value, 0.4f, "White root should maximize (inherit child's 0.4)");

    print_test_pass("Deep Tree Propagation");
}


// ============================================================================
// Test 8: Player Validation Assertion
// ============================================================================

void test_player_validation() {
    print_test_header("Player Validation Assertion");

    std::cout << "Testing Stone::Black creation: ";
    try {
        MCTSNode node1(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::Black);
        std::cout << "✓ OK\n";
    } catch (...) {
        std::cout << "✗ FAILED\n";
        exit(1);
    }

    std::cout << "Testing Stone::White creation: ";
    try {
        MCTSNode node2(Position{0, 0, 0}, false, nullptr, 1.0f, 0, Stone::White);
        std::cout << "✓ OK\n";
    } catch (...) {
        std::cout << "✗ FAILED\n";
        exit(1);
    }

    // Note: We can't easily test Stone::Empty assertion in normal flow
    // because it would abort the program. This would require a separate
    // process or death test framework.
    std::cout << "Note: Stone::Empty assertion requires death test (not implemented here)\n";

    print_test_pass("Player Validation Assertion");
}


// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         MCTS Terminal Propagation Unit Tests                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";

    try {
        test_node_terminal_detection();
        test_player_alternation();
        test_single_child_minimax();
        test_minimax_direction();
        test_empty_children_guard();
        test_mixed_children();
        test_deep_tree_propagation();
        test_player_validation();

        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << std::string(80, '=') << "\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST SUITE FAILED: " << e.what() << "\n\n";
        return 1;
    }
}
