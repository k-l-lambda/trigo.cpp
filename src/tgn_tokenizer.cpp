/**
 * TGN Tokenizer - Implementation
 */

#include "tgn_tokenizer.hpp"
#include <algorithm>


namespace trigo
{


TGNTokenizer::TGNTokenizer()
{
	build_vocab_map();
}


void TGNTokenizer::build_vocab_map()
{
	// Direct identity mapping: token_id = ascii_value
	// ASCII printable: 32 (SPACE) to 127 (DEL)
	for (int ascii_val = 32; ascii_val < 128; ascii_val++)
	{
		byte_to_token_[static_cast<uint8_t>(ascii_val)] = static_cast<int64_t>(ascii_val);
	}

	// Newline for multi-line TGN files
	byte_to_token_[10] = 10;  // LF

	// Token ID -> Byte mapping (inverse)
	for (const auto& [byte, token] : byte_to_token_)
	{
		token_to_byte_[token] = byte;
	}
}


std::vector<int64_t> TGNTokenizer::encode(
	const std::string& text,
	int max_length,
	bool add_special_tokens,
	bool add_value_token,
	bool padding,
	bool truncation
) const
{
	// Convert text to bytes and map to token IDs
	std::vector<int64_t> tokens;
	tokens.reserve(text.size() + 4);  // Reserve space for text + special tokens

	for (unsigned char byte_val : text)
	{
		auto it = byte_to_token_.find(byte_val);
		if (it != byte_to_token_.end())
		{
			tokens.push_back(it->second);
		}
		// Skip out-of-vocabulary bytes (non-ASCII characters)
		// TGN notation should only use mapped ASCII characters
	}

	// Add special tokens
	if (add_value_token)
	{
		// For dual-head networks: [VALUE] [START] ... [END]
		if (add_special_tokens)
		{
			tokens.insert(tokens.begin(), START_ID);
			tokens.insert(tokens.begin(), VALUE_ID);
			tokens.push_back(END_ID);
		}
		else
		{
			tokens.insert(tokens.begin(), VALUE_ID);
		}
	}
	else if (add_special_tokens)
	{
		// Standard: [START] ... [END]
		tokens.insert(tokens.begin(), START_ID);
		tokens.push_back(END_ID);
	}

	// Truncate if needed
	if (truncation && static_cast<int>(tokens.size()) > max_length)
	{
		if (add_special_tokens)
		{
			// Keep START (and VALUE if present), truncate middle, add END
			tokens.resize(max_length - 1);
			tokens.push_back(END_ID);
		}
		else
		{
			tokens.resize(max_length);
		}
	}

	// Pad if needed
	if (padding && static_cast<int>(tokens.size()) < max_length)
	{
		int pad_count = max_length - static_cast<int>(tokens.size());
		tokens.insert(tokens.end(), pad_count, PAD_ID);
	}

	return tokens;
}


std::vector<std::vector<int64_t>> TGNTokenizer::encode_batch(
	const std::vector<std::string>& texts,
	int max_length,
	bool add_special_tokens,
	bool add_value_token,
	bool padding,
	bool truncation
) const
{
	std::vector<std::vector<int64_t>> batch;
	batch.reserve(texts.size());

	for (const auto& text : texts)
	{
		batch.push_back(encode(text, max_length, add_special_tokens,
		                       add_value_token, padding, truncation));
	}

	return batch;
}


std::string TGNTokenizer::decode(
	const std::vector<int64_t>& tokens,
	bool skip_special_tokens
) const
{
	std::vector<uint8_t> byte_values;
	byte_values.reserve(tokens.size());

	for (int64_t token_id : tokens)
	{
		// Skip special tokens
		if (skip_special_tokens && token_id < 8)
		{
			continue;
		}

		// Skip padding
		if (token_id == PAD_ID)
		{
			continue;
		}

		// Convert token to byte
		auto it = token_to_byte_.find(token_id);
		if (it != token_to_byte_.end())
		{
			byte_values.push_back(it->second);
		}
		// else: skip unknown tokens
	}

	// Convert bytes to string
	return std::string(byte_values.begin(), byte_values.end());
}


std::vector<std::string> TGNTokenizer::decode_batch(
	const std::vector<std::vector<int64_t>>& token_batch,
	bool skip_special_tokens
) const
{
	std::vector<std::string> texts;
	texts.reserve(token_batch.size());

	for (const auto& tokens : token_batch)
	{
		texts.push_back(decode(tokens, skip_special_tokens));
	}

	return texts;
}


std::unordered_map<std::string, int> TGNTokenizer::get_special_tokens() const
{
	return {
		{"pad", PAD_ID},
		{"start", START_ID},
		{"end", END_ID},
		{"value", VALUE_ID}
	};
}


}  // namespace trigo
