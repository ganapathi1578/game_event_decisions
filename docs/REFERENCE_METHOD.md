# Reference method implemented

This project independently reimplements the method described by the public `avbiswas/bev-train`
repository and the associated video: https://www.youtube.com/watch?v=sF3CNPbWA8o

Reference repository: https://github.com/avbiswas/bev-train

The reference repository describes the central sequence layout as:

1. Tokenize context/question and each choice separately.
2. Append each choice inside `<option>...</option>` markers.
3. Reset each choice block to the same starting logical position ID.
4. Use a causal context block.
5. Let each choice attend to the entire context and to its own causal block only.
6. Prevent one choice from attending to another choice.
7. Let the final `The answer is:` tokens attend to context and all preceding choices.
8. Read each choice representation at the final token of its `</option>` closing marker (the reference `choices_end_idx`) and read the decision representation at the final token of the answer prompt.
9. Apply the reference-style choice head: task embedding, two or configurable self-attention/MLP residual blocks, and bilinear answer-to-option scores.
10. Train with cross-entropy over option scores, not next-token generation.

The architecture is adapted to SmolVLM2 by obtaining per-frame visual features from the pretrained SmolVLM
vision tower and connector. The context includes native SmolVLM frame prompts and one `<image>` placeholder per
connector token; connector features replace those placeholder embeddings in-place while fake/global image markers
remain in the sequence. The resulting `inputs_embeds`, explicit position IDs, and 4D additive attention mask are
passed into the SmolVLM text backbone. The vision tower and connector are frozen by default. LoRA is applied to the final K text
transformer layers.

This is an independent implementation of the public method's mechanics, not a fork or a claim of exact
parameter-for-parameter reproduction of the creator's Qwen3 experiment. The original uses Qwen3; this
project uses SmolVLM2 and supplies video frame features. Differences required by the backbone are
called out in the README.
