# Lc0 Data Format

This document describes the format of the lc0 training data, which we convert into our own data format.

## V6 Training Data

This is the format of the v6 training data, taken directly from the [lc0 website](https://lczero.org/dev/wiki/training-data-format-versions/).

```cpp
struct V6TrainingData {
  uint32_t version;
  uint32_t input_format;
  float probabilities[1858];
  uint64_t planes[104];
  uint8_t castling_us_ooo;
  uint8_t castling_us_oo;
  uint8_t castling_them_ooo;
  uint8_t castling_them_oo;
  // For input type 3 contains enpassant column as a mask.
  uint8_t side_to_move_or_enpassant;
  uint8_t rule50_count;
  // Bitfield with the following allocation:
  //  bit 7: side to move (input type 3)
  //  bit 6: position marked for deletion by the rescorer (never set by lc0)
  //  bit 5: game adjudicated (v6)
  //  bit 4: max game length exceeded (v6)
  //  bit 3: best_q is for proven best move (v6)
  //  bit 2: transpose transform (input type 3)
  //  bit 1: mirror transform (input type 3)
  //  bit 0: flip transform (input type 3)
  // In versions prior to v5 this spot contained an unused move count field.
  uint8_t invariance_info;
  // In versions prior to v6 this spot contained thr result as an int8_t.
  uint8_t dummy;
  float root_q;
  float best_q;
  float root_d;
  float best_d;
  float root_m;      // In plies.
  float best_m;      // In plies.
  float plies_left;  // This is the training target for MLH.
  float result_q;
  float result_d;
  float played_q;
  float played_d;
  float played_m;
  // The folowing may be NaN if not found in cache.
  float orig_q;      // For value repair.
  float orig_d;
  float orig_m;
  uint32_t visits;
  // Indices in the probabilities array.
  uint16_t played_idx;
  uint16_t best_idx;
  // Kullback-Leibler divergence between visits and policy (denominator)
  float policy_kld;
  uint32_t reserved;
} PACKED_STRUCT;
static_assert(sizeof(V6TrainingData) == 8356, "Wrong struct size");
```
