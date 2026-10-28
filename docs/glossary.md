# Glossary

| Term | Meaning |
|---|---|
| **KSL** | Kenyan Sign Language |
| **Gloss** | Written label for a sign, in CAPITALS (e.g. `WATER`). Not a translation |
| **ISLR / CSLR** | Isolated / Continuous Sign Language Recognition |
| **CTC** | Connectionist Temporal Classification: alignment-free sequence loss with a blank token |
| **MPM** | Masked Pose Modelling: self-supervised pre-training on unlabelled signing |
| **Non-manual markers** | Facial expressions, head and body movements that carry grammatical meaning |
| **Co-articulation** | How adjacent signs change each other's form in continuous signing |
| **Fingerspelling** | Spelling a word letter by letter with handshapes (`FS:JOHN`) |
| **Classifier (CL)** | Handshape representing a class of objects and their movement/location |
| **UtteranceIR** | Our shared internal meaning representation (`ksl_core/ir.py`) |
| **Shallow fusion** | Adding weighted LM log-probabilities during beam search |
| **Context bias** | Small log-probability boosts for venue/dialogue-relevant glosses (≤ 2 nats) |
| **ECE** | Expected Calibration Error |
| **VOI** | Value of information: ask a clarifying question only when it is worth the interruption |
| **Signer-independent split** | No signer appears in more than one of train/val/test |
| **CODA** | Child of Deaf Adults, a hearing native signer |
| **Bootstrap model** | A third-party open-weights model run locally until our own model replaces it |
| **GUC** | PostgreSQL run-time setting (`SET LOCAL app.current_user_id = …`) used by RLS |
| **Medoid clip** | The recording of a sign with the smallest total DTW distance to all others: the dictionary exemplar |
