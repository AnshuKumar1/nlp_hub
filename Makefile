.PHONY: data train eval inference run clean 

data:
	@echo "Creating dataset from google/sentence_compressiom.."
	python -m build_dataset

train:
	@echo "Training google/t5-small model for sentence compression.."
	python -m fine-tuning

eval:
	@echo "Evaluation on test set.."
	python -m utils

inference:
	@echo "Performing model inference on evaluation data.."
	python -m inference

run: clean data train eval inference

clean:
	@find . -name "*.pyc" -exec rm {} \;
	@rm -rf dataset/preprocessed/* checkpoints/* results/*;

