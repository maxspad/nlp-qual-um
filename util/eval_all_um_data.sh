#!/bin/bash
model_max_length=512

echo "QuAL"
python src/um_run_model.py --target_col='qual' \
    --hf_model_name='maxspad/nlp-qual-qual' \
    --hf_model_max_length=$model_max_length
echo

echo "Q1 - Evidence"
python src/um_run_model.py --target_col='evidence' \
    --hf_model_name='maxspad/nlp-qual-q1' \
    --hf_model_max_length=$model_max_length
echo

echo "Q2 - Suggestion"
python src/um_run_model.py --target_col='suggestion' \
    --hf_model_name='maxspad/nlp-qual-q2i' \
    --hf_model_max_length=$model_max_length
echo

echo "Q3 - Connection"
python src/um_run_model.py --target_col='connection' \
    --hf_model_name='maxspad/nlp-qual-q3i' \
    --hf_model_max_length=$model_max_length
