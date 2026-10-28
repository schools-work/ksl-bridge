# KSL gloss conventions (SSOT for annotators, `lexicon.glosses` and the vocabulary)

> Validated against the regex in `lexicon.glosses`: `^[A-Z0-9][A-Z0-9:_.\-]*$`. Only **Deaf reviewers** move a gloss to `approved`.

| Pattern | Meaning | Example |
|---|---|---|
| `WORD` | Lexical sign, English-based label | `WATER`, `WANT` |
| `WORD-WORD` | Multi-word label for one sign | `THANK-YOU` |
| `IX-1`, `IX-2`, `IX-3` | Pointing (index) to self / addressee / third person | `IX-1 WANT WATER` |
| `IX-1-PL`, `IX-3-PL` | Plural pointing | |
| `POSS-1`, `POSS-2` | Possessive | `POSS-1 NAME` |
| `FS:<LETTERS>` | Fingerspelled word | `FS:WANJIKU` |
| `NUM:<n>` | Number sign | `NUM:2` |
| `CL:<CLASS>-<ACTION>` | Classifier construction | `CL:VEHICLE-MOVE` |
| `<GLOSS>.v<n>` | Regional or lexical variant | `EAT.v2` (record `regions` in `lexicon.glosses`) |
| `<GLOSS>+` | Repeated / plural reduplication | `CHILD+` |
| `//` | Utterance boundary in free annotation (not a gloss) | |

**Non-manual tags** (annotated separately, in `annotation.annotations.non_manual`): `yes_no_question`, `wh_question`, `negation`, `intensity`, `topic`. Give each a time span and a 0–1 strength.

**Rules**
1. Gloss what was signed, not the English you would say. `IX-1 WANT BURGER TWO` is a gloss sequence; "I want two burgers" is the translation.
2. Don't drop signs you're unsure about. Gloss them with `?` in the free-text note and flag the task for adjudication.
3. Never "correct" a regional variant to the Nairobi form. Create or choose the variant gloss.
4. Directional or spatial signs: set `is_directional = true` so augmentation never mirrors them.
5. Critical vocabulary (pain, allergy, emergency, medication): set `is_critical = true`.
