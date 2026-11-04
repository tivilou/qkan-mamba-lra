# MEGA adaptation boundary

Retain official MultiHeadEMA, MovingAverageGatedAttention, activation, relative bias, normalization and feedforward modules. Freeze all constructor choices in the eventual task config. Common-protocol classification/pair heads and pixel transforms require explicit source-vs-adapter forward/loss/gradient/update evidence.

Author Image normalization is mean=.48/std=.24; Pathfinder mean=std=.5, differing from project x/255-.5. Do not confuse author-recipe reproduction with common-input comparison. BatchNorm statistics depend on physical micro-batches; gradient accumulation alone does not establish full-batch equivalence. A normalization change must be declared and separately checked.

Native training must save model buffers, optimizer groups, scheduler, all RNGs, committed sampler cursor, epoch aggregates and best-selection state atomically; test only selected validation weights. Future trace capture must preserve exact model-visible inputs, lengths/masks and representations. Raw/derived task data stay outside exchange.
