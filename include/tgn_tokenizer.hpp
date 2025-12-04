/**
 * TGN Tokenizer - C++ implementation compatible with Python TGNTokenizer
 *
 * This tokenizer uses a vocabulary of 128 tokens with direct ASCII mapping:
 * - 0-3: Special tokens (PAD, START, END, VALUE)
 * - 4-7: Reserved for future use
 * - 10: Newline (LF)
 * - 32-127: ASCII printable characters (direct identity mapping)
 *
 * Design principles:
 * 1. Memory efficiency: Reduced vocabulary (128 vs 259)
 * 2. Simplicity: Token ID = ASCII value (no complex mapping)
 * 3. TGN compatibility: All ASCII characters directly accessible
 * 4. Value head support: VALUE token for dual-head networks
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>


namespace trigo
{


class TGNTokenizer
{
public:
	// Vocabulary size
	static constexpr int VOCAB_SIZE = 128;

	// Special tokens (0-3 used, 4-7 reserved)
	static constexpr int PAD_ID = 0;
	static constexpr int START_ID = 1;
	static constexpr int END_ID = 2;
	static constexpr int VALUE_ID = 3;  // For value evaluation in dual-head networks


	/**
	 * Constructor
	 */
	TGNTokenizer();


	/**
	 * Encode text to token IDs.
	 *
	 * @param text Input text to tokenize
	 * @param max_length Maximum sequence length
	 * @param add_special_tokens Add START and END tokens
	 * @param add_value_token Add VALUE token at beginning (for value head)
	 * @param padding Pad sequences to max_length
	 * @param truncation Truncate sequences exceeding max_length
	 * @return Vector of token IDs
	 *
	 * Example:
	 *     TGNTokenizer tokenizer;
	 *     // Regular encoding
	 *     auto tokens = tokenizer.encode("B3 000");
	 *     // With value token (for dual-head training)
	 *     auto tokens = tokenizer.encode("B3 000", 2048, true, true);
	 *     // Result: [VALUE_ID, START_ID, ...tokens..., END_ID, PAD, PAD, ...]
	 */
	std::vector<int64_t> encode(
		const std::string& text,
		int max_length = 2048,
		bool add_special_tokens = true,
		bool add_value_token = false,
		bool padding = true,
		bool truncation = true
	) const;


	/**
	 * Encode multiple texts to batch of token vectors.
	 *
	 * @param texts List of input texts
	 * @param max_length Maximum sequence length
	 * @param add_special_tokens Add START and END tokens
	 * @param add_value_token Add VALUE token at beginning
	 * @param padding Pad sequences to max_length
	 * @param truncation Truncate sequences exceeding max_length
	 * @return Vector of token ID vectors [batch_size][max_length]
	 */
	std::vector<std::vector<int64_t>> encode_batch(
		const std::vector<std::string>& texts,
		int max_length = 2048,
		bool add_special_tokens = true,
		bool add_value_token = false,
		bool padding = true,
		bool truncation = true
	) const;


	/**
	 * Decode token IDs back to text.
	 *
	 * @param tokens Token IDs to decode
	 * @param skip_special_tokens Skip special tokens (0-7)
	 * @return Decoded text string
	 */
	std::string decode(
		const std::vector<int64_t>& tokens,
		bool skip_special_tokens = true
	) const;


	/**
	 * Decode batch of token vectors to texts.
	 *
	 * @param token_batch Batch of token IDs [batch_size][seq_len]
	 * @param skip_special_tokens Skip special tokens (0-7)
	 * @return List of decoded text strings
	 */
	std::vector<std::string> decode_batch(
		const std::vector<std::vector<int64_t>>& token_batch,
		bool skip_special_tokens = true
	) const;


	/**
	 * Get vocabulary size.
	 *
	 * @return Vocabulary size (128)
	 */
	int get_vocab_size() const
	{
		return VOCAB_SIZE;
	}


	/**
	 * Get special token IDs.
	 *
	 * @return Map of token name to token ID
	 */
	std::unordered_map<std::string, int> get_special_tokens() const;


private:
	/**
	 * Build bidirectional mapping between bytes and token IDs.
	 * Uses direct identity mapping: token_id = ascii_value
	 */
	void build_vocab_map();


	// Byte to token ID mapping (ASCII value -> token ID)
	std::unordered_map<uint8_t, int64_t> byte_to_token_;

	// Token ID to byte mapping (token ID -> ASCII value)
	std::unordered_map<int64_t, uint8_t> token_to_byte_;
};


}  // namespace trigo
