/**
 * Test TGN coordinate system
 */

#include "../include/trigo_types.hpp"
#include "../include/trigo_coords.hpp"
#include <iostream>
#include <cassert>
#include <vector>


using namespace trigo;


void test_encode_decode_basic()
{
	std::cout << "\n[TEST] Basic Encode/Decode\n";
	std::cout << "===========================\n";

	BoardShape shape(5, 5, 5);

	// Test center
	Position center(2, 2, 2);
	std::string encoded = encode_ab0yz(center, shape);
	std::cout << "Center (2,2,2) → \"" << encoded << "\"\n";
	assert(encoded == "000");

	Position decoded = decode_ab0yz_position(encoded, shape);
	assert(decoded == center);
	std::cout << "✓ Center encode/decode passed\n";

	// Test corner
	Position corner(0, 0, 0);
	encoded = encode_ab0yz(corner, shape);
	std::cout << "Corner (0,0,0) → \"" << encoded << "\"\n";
	assert(encoded == "aaa");

	decoded = decode_ab0yz_position(encoded, shape);
	assert(decoded == corner);
	std::cout << "✓ Corner encode/decode passed\n";

	// Test opposite corner
	Position opposite(4, 4, 4);
	encoded = encode_ab0yz(opposite, shape);
	std::cout << "Opposite corner (4,4,4) → \"" << encoded << "\"\n";
	assert(encoded == "zzz");

	decoded = decode_ab0yz_position(encoded, shape);
	assert(decoded == opposite);
	std::cout << "✓ Opposite corner encode/decode passed\n";

	// Test mixed
	Position mixed(0, 2, 4);
	encoded = encode_ab0yz(mixed, shape);
	std::cout << "Mixed (0,2,4) → \"" << encoded << "\"\n";
	assert(encoded == "a0z");

	decoded = decode_ab0yz_position(encoded, shape);
	assert(decoded == mixed);
	std::cout << "✓ Mixed encode/decode passed\n";
}


void test_2d_board_compaction()
{
	std::cout << "\n[TEST] 2D Board Compaction\n";
	std::cout << "==========================\n";

	BoardShape shape2d(19, 19, 1);

	// For 2D boards, trailing 1 is ignored
	Position pos2d(0, 0, 0);
	std::string encoded = encode_ab0yz(pos2d, shape2d);
	std::cout << "2D Board (0,0,0) → \"" << encoded << "\"\n";
	assert(encoded == "aa");  // Not "aa0"

	Position decoded = decode_ab0yz_position(encoded, shape2d);
	assert(decoded == pos2d);
	std::cout << "✓ 2D board compaction passed\n";

	// Center of 19x19 board
	Position center(9, 9, 0);
	encoded = encode_ab0yz(center, shape2d);
	std::cout << "2D Board center (9,9,0) → \"" << encoded << "\"\n";
	assert(encoded == "00");

	decoded = decode_ab0yz_position(encoded, shape2d);
	assert(decoded == center);
	std::cout << "✓ 2D board center passed\n";
}


void test_roundtrip_all_positions()
{
	std::cout << "\n[TEST] Round-trip All Positions (5x5x5)\n";
	std::cout << "========================================\n";

	BoardShape shape(5, 5, 5);
	int total = 0;
	int passed = 0;

	for (int x = 0; x < 5; x++)
	{
		for (int y = 0; y < 5; y++)
		{
			for (int z = 0; z < 5; z++)
			{
				Position pos(x, y, z);
				std::string encoded = encode_ab0yz(pos, shape);
				Position decoded = decode_ab0yz_position(encoded, shape);

				total++;
				if (decoded == pos)
				{
					passed++;
				}
				else
				{
					std::cout << "✗ Failed: (" << x << "," << y << "," << z << ") → \""
					          << encoded << "\" → (" << decoded.x << "," << decoded.y << "," << decoded.z << ")\n";
				}
			}
		}
	}

	std::cout << "Round-trip test: " << passed << "/" << total << " passed\n";
	assert(passed == total);
	std::cout << "✓ All positions round-trip correctly\n";
}


void test_error_handling()
{
	std::cout << "\n[TEST] Error Handling\n";
	std::cout << "=====================\n";

	BoardShape shape(5, 5, 5);

	// Invalid length
	bool caught = false;
	try
	{
		decode_ab0yz_position("00", shape);  // Too short
	}
	catch (const std::invalid_argument& e)
	{
		std::cout << "✓ Caught invalid length: " << e.what() << "\n";
		caught = true;
	}
	assert(caught);

	// Invalid character
	caught = false;
	try
	{
		decode_ab0yz_position("0X0", shape);  // 'X' is invalid
	}
	catch (const std::invalid_argument& e)
	{
		std::cout << "✓ Caught invalid character: " << e.what() << "\n";
		caught = true;
	}
	assert(caught);

	std::cout << "✓ Error handling tests passed\n";
}


int main()
{
	std::cout << "TGN Coordinate System Test Suite\n";
	std::cout << "=================================\n";

	try
	{
		test_encode_decode_basic();
		test_2d_board_compaction();
		test_roundtrip_all_positions();
		test_error_handling();

		std::cout << "\n" << std::string(70, '=') << "\n";
		std::cout << "✅ ALL TESTS PASSED!\n";
		std::cout << "TGN coordinate system is working correctly.\n";
		std::cout << std::string(70, '=') << "\n";

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
		return 1;
	}
}
